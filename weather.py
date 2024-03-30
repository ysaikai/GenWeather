"""
Generating weather for the selected scenario (1 or 2). Give an argument
when executing the script, e.g. `python weather 2`.
Assumed `.pt` files to load trained neural networks.
"""

# %%
import torch
from torch.distributions.gamma import Gamma
import numpy as np
import pandas as pd
import sys, os, pathlib
from datetime import datetime, date, timedelta
from glob import glob
from library import Net
import shutil
from copy import deepcopy
from tabulate import tabulate


# %%
"""Parameters"""
# Scenario
try:
  scenario = int(float(sys.argv[1]))
  if scenario not in (1,2):
    raise Exception()
except:
  print("Scenario should be 1 or 2.")
  scenario = 1
print("scenario =", scenario)

n_MC = 1000
years = (2021, 2022, 2023)
if scenario==1:
  model_loaded = "scenario1.pt"
  n_years = 1
elif scenario==2:
  model_loaded = "scenario2.pt"
  n_years = 3
dir_name = "met files" + str(scenario)
n_pred = 365*n_years


"""Data"""
# train.csv & header.met are generated in train.py
df_in = pd.read_csv("train.csv", sep="\s+")
df_in = df_in.sort_values(by=["year","day"])
# Training set (c.f. the same line in ~L.135, train.py)
df_train = df_in.iloc[:-(365*n_years)].drop(columns=["year","day"])
mean = np.array(df_train.mean())
std = np.array(df_train.std())

# Latest date
with open("train.met", "r") as f:
  # Split the whitespace-separated line
  l = f.readlines()[-1].split()
  # 1st is year and 2nd is day of year
  date_latest = date(int(l[0]),1,1) + timedelta(int(l[1])-1)

cols = ["radn", "mint", "diff", "rain"] # used in the training
cols0 = ["radn", "maxt", "mint", "rain"] # original column names
cols1 = ["radn", "mint", "maxt", "rain"] # mint & maxt switched
cols_tex = ["Period", "Radn", "MinT", "MaxT", "Rain"]


# %%
"""Init"""
torch.manual_seed(1)

# Model
print("Loaded PyTorch model:", model_loaded)
model = Net(n_pred)
n_cond = model.n_cond
height = model.height
size_mask = model.size_mask

model = torch.nn.DataParallel(model, device_ids=model.device_ids)
model.load_state_dict(torch.load(model_loaded, map_location="cpu"))
model = model.module.to(torch.device("cpu"))
model.eval()


# %%
"""Weather sampling"""
# Clear the directory
try:
  shutil.rmtree(dir_name)
except:
  pass
os.makedirs(dir_name, exist_ok=True)
with open("header.met", "r") as f: # Header information for .met files
  header = f.readlines()


def sample(m, n_pred, x0, date0):
  """
  PyTorch models take an input in NCHW format by default:
    N: size of batch
    C: number of channels
    (H,W: height & width in 2-D input)
  """
  out = [] # data for df_out
  x_test = torch.tensor(x0, dtype=torch.float32)
  for t in range(n_pred):
    """
    For the masked conv, shift the top row of 0s to the bottom, which is
    sequantially replaced with sampled values.
    """
    x_test = torch.cat((x_test[1:], torch.zeros((1,height))))

    for i in range(height):
      with torch.no_grad():
        # Transpose, add a batch dimension, and add a channel dimension
        a = model(x_test.transpose(0,1).unsqueeze(0).unsqueeze(0), i)

      if i == 1:
        # MinT (Gaussian)
        y_pred[m][t][i] = torch.normal(*a[0,0])
      else:
        # Radn, MaxT-MinT, Rain (gamma)
        y_pred[m][t][i] = Gamma(*a[0,0]).sample()
      
      x_test[-1][i] = (y_pred[m][t][i] - mean[i])/std[i] # replace 0

    # Update df_out for .met file export
    _d = date0 + timedelta(t)
    out.append([_d.year, _d.timetuple().tm_yday] + list(y_pred[m][t]))
  df_out = pd.DataFrame(out, columns=["year","day"]+cols)
  
  # Write a met file
  df_out["diff"] += df_out["mint"] # Diff -> MaxT
  df_out[cols] = df_out[cols].round(2) # Round
  df_out = df_out.rename(columns={"diff": "maxt"}) # Rename
  df_out = df_out[["year","day"] + cols0] # Reorder
  df_out = df_out.astype({"year": "int", "day": "int"})
  filename = "tmp" + str(scenario) + ".csv"
  df_out.to_csv(filename, sep=" ", index=False, header=False)
  with open(filename, "r") as f:
    out = header + f.readlines()
  if scenario == 1:
    filename = "{}/s{}_{:05d}.met".format(dir_name, str(year), m)
  elif scenario == 2:
    filename = "{}/r{:05d}.met".format(dir_name, m)
  with open(filename, "w", newline="\r\n") as f:
    f.writelines(out)


# %%
"""Sampling"""
cnt = 0
if scenario == 1:
  cnt_tot = n_MC*len(years)
  for year in years:
    y_pred = np.zeros((n_MC, n_pred, height))
    # First day of the prediction window. Hence -1 in timedelta()
    date0 = date_latest - timedelta(n_pred*(years[-1]-year+1)-1)
    row_date0 = df_in[(df_in["year"]==date0.year) &
                      (df_in["day"]==date0.timetuple().tm_yday)].index[0]
    _df = df_in.iloc[:row_date0]
    # len(x)=n_cond despite the masked conv. 1 added in `sample()`.
    _df = _df.iloc[-n_cond:][cols]
    x0 = (_df.values - mean)/std
    """
    Compared with ZeroPad1d(((self.n_pred-1)+(size_mask-1-1), 0)) used in
    the training, 1 more row of 0s padded as the top row of 0s is shifted to
    the bottom in sample().
    """
    n_rows = (n_pred-1)+(size_mask-1) # not .size_mask-1-1
    x0 = np.vstack((np.zeros((n_rows,x0.shape[1])), x0))
    for m in range(n_MC):
      sample(m, n_pred, x0, date0)
      cnt += 1
      if cnt%(max(1,cnt_tot//10))==0:
        print("{}/{} Year {}".format(m, n_MC-1, year))
elif scenario == 2:
  y_pred = np.zeros((n_MC, n_pred, height))
  cnt_tot = n_MC
  # First day of the prediction window. Hence -1 in timedelta()
  date0 = date_latest - timedelta(n_pred-1)
  row_date0 = df_in[(df_in["year"]==date0.year) &
                    (df_in["day"]==date0.timetuple().tm_yday)].index[0]
  _df = df_in.iloc[:row_date0]
  # len(x)=n_cond despite the masked conv. 1 added in `sample()`.
  _df = _df.iloc[-n_cond:][cols]
  x0 = (_df.values - mean)/std
  n_rows = (n_pred-1)+(size_mask-1)
  x0 = np.vstack((np.zeros((n_rows,x0.shape[1])), x0))
  for m in range(n_MC):
    sample(m, n_pred, x0, date0)
    cnt += 1
    if cnt%(max(1,cnt_tot//10))==0:
      print("{}/{}".format(m, n_MC-1))

filename = "tmp" + str(scenario) + ".csv"
os.remove(filename) # Delete a tmp file


# %%
"""Postprocess"""
def postprocess(year, dfs):
  y_tex = np.zeros((len(dfs), n_pred, len(cols1)))
  for i, df in enumerate(dfs):
    y_tex[i] = df[:n_pred][cols1].values

  out = [] # data for df_tex

  # date0 is the first day of the prediction window.
  df_true = df_in.iloc[row_date0:row_date0+n_pred].reset_index(drop=True)
  df_true["diff"] += df_true["mint"] # Diff -> MaxT
  df_true = df_true.rename(columns={"diff": "maxt"}) # Rename
  headers = [""] + cols_tex[1:]

  # Daily
  table = []
  y_true = df_true[cols1].values
  err = abs(y_tex - y_true).mean(axis=(0,1))
  table.append(["Daily"] + list(err))
  table.append([])
  print(tabulate(table, headers, floatfmt=".2f"))

  # Weekly
  table = []
  err = np.zeros((52,len(cols1)))
  days = [7]*52
  for i in range(52):
    idx = df_true[(df_true["day"]>=1+sum(days[0:i])) &
                  (df_true["day"]<=sum(days[0:i+1]))].index
    err[i] = abs(y_tex[:,idx,:].mean(axis=1) -\
                 df_true.iloc[idx][cols1].values.mean(axis=0)).mean(axis=0)
  table.append(["Weekly"] + list(err.mean(axis=0)))
  table.append([])
  ## Details
  for i in range(52):
    week = "Week {:02d}".format(i+1)
    row = [week] + list(err[i].round(2))
    table.append(row)
    out.append(row) # for df_tex
  table.append([])
  print(tabulate(table, headers, floatfmt=".2f"))

  # Monthly
  table = []
  err = np.zeros((12,len(cols1)))
  days = [(date(year,m+1,1)-date(year,m,1)).days for m in range(1,12)]+[31]
  for i in range(12):
    # Get the indices for the (i+1)th month
    idx = df_true[(df_true["day"]>=1+sum(days[0:i])) &
                  (df_true["day"]<=sum(days[0:i+1]))].index
    err[i] = abs(y_tex[:,idx,:].mean(axis=1) -\
                 df_true.iloc[idx][cols1].values.mean(axis=0)).mean(axis=0)
  table.append(["Monthly"] + list(err.mean(axis=0)))
  table.append([])
  for i in range(12):
    month = date(year,i+1,1).strftime("%B")
    row = [month] + list(err[i].round(2))
    table.append(row)
    out.append(row) # for df_tex
  table.append([])
  print(tabulate(table, headers, floatfmt=".2f"))

  # TeX file
  df_tex = pd.DataFrame(out, columns=cols_tex)
  if scenario == 1:
    filename = "weather1_{}.tex".format(year)
  elif scenario == 2:
    filename = "weather2.tex"
  df_tex.to_latex(filename, index=False, float_format="%g")
  with open(filename, "r") as f:
    lines = f.readlines()
  with open(filename, "w") as f:
    for line in lines:
      if not ("rule" in line):
        line = line.replace("\\\\\n", "\\\\\\hline\n")
        f.write(line)


met_files = glob(dir_name + "/*.met")
if scenario == 1:
  dfs = {yr:[] for yr in years}
  for met_file in met_files:
    year = int(met_file[-14:-10])
    with open(met_file, "r") as f:
      lines = f.readlines()
    with open("tmp.csv", "w") as f:
      f.writelines(lines[22:23] + lines[24:]) # L23 and L25-
    df = pd.read_csv("tmp.csv", sep="\s+")
    # Reorder the columns
    dfs[year].append(df[["year","day"] + cols1])

  for year in years:
    print(year)
    # First day of the prediction window. Hence -1 in timedelta()
    date0 = date_latest - timedelta(n_pred*(years[-1]-year+1)-1)
    row_date0 = df_in[(df_in["year"]==date0.year) &
                      (df_in["day"]==date0.timetuple().tm_yday)].index[0]
    postprocess(year, dfs[year])

elif scenario == 2:
  dfs = []
  year = years[0]
  for met_file in met_files:
    with open(met_file, "r") as f:
      lines = f.readlines()
    with open("tmp.csv", "w") as f:
      f.writelines(lines[22:23] + lines[24:]) # L23 and L25-
    df = pd.read_csv("tmp.csv", sep="\s+")
    # Reorder the columns
    dfs.append(df[["year","day"] + cols1])

  # First day of the prediction window. Hence -1 in timedelta()
  date0 = date_latest - timedelta(n_pred*(years[-1]-year+1)-1)
  row_date0 = df_in[(df_in["year"]==date0.year) &
                    (df_in["day"]==date0.timetuple().tm_yday)].index[0]
  postprocess(year, dfs)


# %%
