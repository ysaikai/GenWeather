# %%
import torch
from torch.utils.data import DataLoader, Dataset
from torch import lgamma
import numpy as np
import pandas as pd
import sys, os, pathlib
from datetime import datetime, date, timedelta
from library import Net
from time import monotonic

torch.manual_seed(1)


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
n_years = 1 + 2*(scenario-1) # either 1 or 3
size_batch = 1000
n_epoch = 2000
device_ids = [0]
EPS = torch.tensor(1e-3) # for stability
model_loaded = ""
print("Scenario={}, model_loaded={}".format(scenario, model_loaded))


"""Init"""
# Device
if torch.cuda.is_available():
  device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
  device = torch.device("mps")
else:
  device = torch.device("cpu")
  device_ids = []

# Model
model = Net(365*n_years, 4, device_ids)
n_par = sum(
	param.numel() for param in model.parameters()
)
n_rcpt = model.n_rcpt
n_cond = model.n_cond
height = model.height
out_channels = model.out_channels

model = torch.nn.DataParallel(model, device_ids=device_ids)
if model_loaded != "":
  model.load_state_dict(torch.load(model_loaded, map_location=device))
model.to(device)


# Negative log-likelihood
def loss_fn(a, y):
  y = y.transpose(-1,-2) # Swap the last two dims

  # MinT (Gaussian)
  p1 = a[1][:,:,0]
  p2 = a[1][:,:,1]
  l = torch.log(p2) + 0.5*((y[:,:,1]-p1)/p2)**2

  # MaxT-MinT, Radn, Rain (gamma)
  for i in [x for x in range(height) if x!=1]:
    p1 = a[i][:,:,0]
    p2 = a[i][:,:,1]
    l += -p1*torch.log(p2)+lgamma(p1)-(p1-1)*torch.log(y[:,:,i])+p2*y[:,:,i]

  return torch.mean(l)


# %%
"""Data
For some reasons, Pandas doesn't like to directry read .met files
downloaded from SILO. So, save it as a temporary file and read it.

It is sensitive to the structure of .met files from SILO. In Robe.met,
  -L1-3, 8-22. comments indicated by ! symbol at the beginning
  -L4-7. lon & lat and two ave temperatures, necessary for running
  -L23. header for column names
  -L24. units of the header items
L23 and L25- are used to create a dataframe.

train.met the same as the original .met file without the last three
["evap", "vp", "code"] columns as irrelevant. It is used for benchmarking
in apsim#.py.

train.csv is a modified version of the original dataset, i.e.
  - maxt -> diff
  - the column order of [radn, mint, diff, rain]

`date_latest` is currently not used, but will be useful when it falls
in the middle of season and we won't simply create a traingn set
by excluding 365*3 days from the latest date.
"""
filename = "Robe.met"
cols = ["year", "day", "radn", "mint", "diff", "rain"]
cols0 = ["year", "day", "radn", "maxt", "mint", "rain"] # for train.met

# Latest date
with open(filename, "r") as f:
  # Split the whitespace-separated line
  l = f.readlines()[-1].split()
  # 1st is year and 2nd is day of year
  date_latest = date(int(l[0]),1,1) + timedelta(int(l[1])-1)

# Pre-process .met file
with open(filename, "r") as f:
  lines = f.readlines()
  header = lines[:22] # To which the generated weather data is appended
  header.append("year day radn maxt mint rain\n")
  header.append("() () (MJ/m^2) (oC) (oC) (mm)\n")
with open("header.met", "w", newline="\r\n") as f:
  f.writelines(header)
with open("train.csv", "w") as f:
  f.writelines(lines[22:23] + lines[24:]) # L23 and L25-
df_in = pd.read_csv("train.csv", sep="\s+", dtype=np.float32)
df_in = df_in.drop(columns=["evap", "vp", "code"]) # irrelevant columns
df_in = df_in.astype({"year": "int", "day": "int"})

# train.met
df_in[cols0].to_csv("train.met", sep=" ", index=False)
with open("train.met", "r") as f:
  out = header + f.readlines()[1:]
with open("train.met", "w", newline="\r\n") as f: # for Windows
  f.writelines(out)

# train.csv 
df_in["maxt"] = (df_in["maxt"] - df_in["mint"]).round(2) # MaxT - MinT
df_in = df_in.rename(columns={"maxt": "diff"}) # Rename
df_in = df_in[cols] # reorder the columns
df_in[["diff","radn","rain"]] = np.maximum(EPS, df_in[["diff","radn","rain"]])
df_in = df_in.sort_values(by=["year","day"])
df_in.to_csv("train.csv", sep=" ", index=False)


# %%
"""Training dataset
It excludes the last 365*3 entries, which is the size of test set for
Scnerio 2. For Scenario 1, we assume regardless of the test year
2021, 2022, or 2023 season, the training is carried out usign the same
dataset.

len(x)=n_rcpt+1 due to the mask conv, which needs one more time step.
It doesn't alter len(y)=n_pred=n_rcpt-n_cond+1.
"""
df_train = df_in.iloc[:-(365*3)].drop(columns=["year","day"])
mean = np.array(df_train.mean())
std = np.array(df_train.std())

X = []
Y = []
for i in range(df_train.shape[0] - n_rcpt):
  x = (df_train.iloc[i:i+n_rcpt+1] - mean)/std
  X.append(np.expand_dims(x, axis=0))
  y = df_train.iloc[i+n_cond:i+n_rcpt+1].values
  Y.append(y)
# Transpose so that the temporal dimensiona comes last
X = torch.tensor(np.array(X)).transpose(-1,-2)
Y = torch.tensor(np.array(Y), dtype=torch.float32).transpose(-1,-2)
ds = torch.utils.data.TensorDataset(X, Y)
dl = DataLoader(ds, size_batch, shuffle=True, drop_last=True)


# %%
"""Run"""
msg = "{} | #epoch {} | #batch {} | #par {} | #channels {}"
print(msg.format(device,n_epoch, size_batch, n_par, out_channels))
optimiser = torch.optim.Adam(model.parameters())
loss_min = np.inf
name_pt = "init.pt"
for epoch in range(n_epoch):
  stamp = monotonic() # timestamp
  for x, y in dl:
    x = x.to(device)
    y = y.to(device)
    a = model(x)
    loss = loss_fn(a, y)
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

  # Quit if NaN
  if torch.isnan(loss):
    torch.save(model.state_dict(), "NaN.pt")
    sys.exit(0)

  # Remember the best
  if loss.item() < loss_min:
    # Delete the existing .pt
    pathlib.Path(name_pt).unlink(missing_ok=True)
    # Update the loss & filename
    loss_min = loss.item()
    name_pt = "loss{:.2f}({:d}-{:d}).pt".format(loss_min, epoch, n_epoch)
    # Save .pt
    torch.save(model.state_dict(), name_pt)

  # Progress
  if epoch%(max(1,n_epoch//100))==0:
    if torch.cuda.is_available():
      msg = "{:05d} Loss={:.2f} ({:.1f}sec/epoch, {:.1f}/{:.1f}GB)"
      mem = torch.cuda.mem_get_info()
      use = (mem[1]-mem[0])/2**30
      print(msg.format(epoch,loss.item(),monotonic()-stamp, use, mem[1]/2**30))
    else:
      msg = "{:05d} Loss={:.2f} ({:.1f}sec/epoch)"
      print(msg.format(epoch,loss.item(),monotonic()-stamp))

# Save the the last as well
name_pt = "{:%Y%m%d%H%M}.pt".format(datetime.now())
torch.save(model.state_dict(), name_pt) # Last model
print("{}, loss={:.2f}".format(name_pt, loss.item()))


# %%
