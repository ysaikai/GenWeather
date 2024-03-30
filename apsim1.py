"""
APSIM runs for Scenario 1

Benchmark and Truth use train.met, which is essentially the same as
the original dataset and prepared in train.py.
"""

# %%
import numpy as np
import pandas as pd
import sys, os, copy, json, subprocess
from datetime import datetime, date, timedelta
from glob import glob
import shutil
from tabulate import tabulate
import sqlite3

os.chdir(os.path.dirname(os.path.abspath(__file__)))


# %%
"""Parameters"""
path_apsim = "C:/Program Files/APSIM2023.11.7349.0/bin/Models.exe"
years = [2021, 2022, 2023]
crops = ["wheat", "barley", "canola"]
dir_name = "scenario1"
k = "Children" # dictionary key
met_files = glob("met files1/*.met")
n_MC = int(len(met_files)/len(years))

try:
  shutil.rmtree(dir_name) # Clear the directory
except:
  pass
os.makedirs(dir_name, exist_ok=True)


# %%
"""Preprocess"""
# Get the start months and end days
month_start = {}
day_start = {}
for year in years:
  with open("met files1/s" + str(year) + "_00000.met", "r") as f:
    lines = f.readlines()
    l = lines[24].split() # Make sure L.25 is the 1st line
    date_start = date(int(l[0]),1,1) + timedelta(int(l[1])-1)
    month_start[year] = date_start.month
    day_start[year] = date_start.day

# Set the reporting month & day (independent of the year, so far)
## n.b. Using the fixed date because, if using EndOfSimulation, APSIM
## peculiarly resets soil water before outputing it :(
date_end = date_start + timedelta(364)
for crop in crops:
  with open(crop + ".apsimx", "r") as f:
    data = json.load(f)
  data[k][1][k][1][k][5][k][0]["EventNames"][1] = date_end.strftime("%d-%b")
  data = json.dumps(data, indent=2)
  path = crop + ".apsimx"
  with open(path, "w") as f:
    f.write(data)


# %%
"""Generated weather"""
print("Gen")
for crop in crops:
  print(crop)
  with open(crop + ".apsimx", "r") as f:
    data = json.load(f)

  x = copy.deepcopy(data[k][1])
  for _ in range(2):
    x = x[k][0]
  x = x[k]

  for year in years:
    for i in range(n_MC):
      x.append(copy.deepcopy(x[0]))
      name = "{}_{:05d}".format(year, i)
      x[-1]["Name"] = name
      x[-1][k][0]["FileName"] = "..\\met files1\\s" + name + ".met"
      x[-1][k][0]["Name"] = name
      date_start = date(year, month_start[year], day_start[year])
      x[-1]["Specifications"][0] = "[Clock].StartDate = " +\
                                  date_start.strftime("%Y-%m-%d")
      date_end = date_start + timedelta(364)
      x[-1]["Specifications"][1] = "[Clock].EndDate = " +\
                                  date_end.strftime("%Y-%m-%d")
  del x[0] # Delete the duplicated first element

  data[k][1][k][0][k][0][k] = x
  data = json.dumps(data, indent=2)
  path = dir_name + "/" + crop + "_g.apsimx"
  with open(path, "w") as f:
    f.write(data)

  subprocess.run([path_apsim, path])


# %%
"""Benchmark"""
print("Benchmark")
rng = np.random.default_rng(1) # Fix a random seed
for crop in crops:
  print(crop)
  with open(crop + ".apsimx", "r") as f:
    data = json.load(f)

  x = copy.deepcopy(data[k][1])
  for _ in range(2):
    x = x[k][0]
  x = x[k]

  for year in years:
    yrs = rng.integers(low=year-30, high=year, size=n_MC)
    for i, yr in enumerate(yrs):
      x.append(copy.deepcopy(x[0]))
      name = "{}_{:05d}".format(year, i)
      x[-1]["Name"] = name
      x[-1][k][0]["FileName"] = "..\\train.met" # default name
      x[-1][k][0]["Name"] = name
      # Any year works to get month_start & day_start
      date_start = date(yr, month_start[year], day_start[year])
      x[-1]["Specifications"][0] = "[Clock].StartDate = " +\
                                   date_start.strftime("%Y-%m-%d")
      date_end = date_start + timedelta(364)
      x[-1]["Specifications"][1] = "[Clock].EndDate = " +\
                                   date_end.strftime("%Y-%m-%d")

  del x[0] # Delete the duplicated first element

  data[k][1][k][0][k][0][k] = x
  data = json.dumps(data, indent=2)
  path = dir_name + "/" + crop + "_b.apsimx"
  with open(path, "w") as f:
    f.write(data)

  subprocess.run([path_apsim, path])


# %%
"""Truth"""
print("Truth")
for crop in crops:
  with open(crop + ".apsimx", "r") as f:
    data = json.load(f)

  x = copy.deepcopy(data[k][1])
  for _ in range(2):
    x = x[k][0]
  x = x[k]

  for year in years:
    x.append(copy.deepcopy(x[0]))
    # name = "{}".format(year)
    name = "{}_{:05d}".format(year, 0)
    x[-1]["Name"] = name
    x[-1][k][0]["FileName"] = "..\\train.met" # default name
    x[-1][k][0]["Name"] = name
    date_start = date(year, month_start[year], day_start[year])
    x[-1]["Specifications"][0] = "[Clock].StartDate = " +\
                                  date_start.strftime("%Y-%m-%d")
    date_end = date_start+timedelta(364)
    x[-1]["Specifications"][1] = "[Clock].EndDate = " +\
                                  date_end.strftime("%Y-%m-%d")

  del x[0] # Delete the duplicated first element

  data[k][1][k][0][k][0][k] = x
  data = json.dumps(data, indent=2)
  path = dir_name + "/" + crop + "_t.apsimx"
  with open(path, "w") as f:
    f.write(data)

  subprocess.run([path_apsim, path])


# %%
"""Read results .db and put them in df_out
SQL refresher
  - "SELECT name FROM sqlite_master WHERE type='table'" # list of tables
  - "PRAGMA table_info(Report)" # list of column names
  - Use `` to escape . used in the query
"""
q = "SELECT `Clock.Today`, WeatherYear, Yield"
for i in range(1,7): # Assumed 6 soil layers
  q += ", `Soil.Water.MM({})`".format(i)
q +=  " FROM Report ORDER BY WeatherYear"
cols = ["year", "method", "crop", "yield", "water"]
data = []
for method in ("g", "b", "t"):
  for crop in crops:
    con = sqlite3.connect(dir_name + "/" + crop + "_" + method + ".db")
    cur = con.cursor()
    re = cur.execute(q)

    """Assumed yield=0 on the end date"""
    year = 1000
    idx = -1
    for row in re.fetchall():
      # print(row)
      if int(row[1][:4]) != year or int(row[1][5:]) != idx:
        y = None
        sw = None
        year = int(row[1][:4])
        idx = int(row[1][5:])
      d = datetime.strptime(row[0][:10], "%Y-%m-%d").date()
      if float(row[2]) > 0:
        y = float(row[2])
      if d.month == date_end.month and d.day == date_end.day:
        sw = sum(row[3:9])
      if y is not None and sw is not None:
        data.append([int(row[1][:4]), method, crop, y, sw])
df_out = pd.DataFrame(data, columns=cols)


# %%
"""Summarise df_out in terms of errors per crop/method/year"""
cols_y = ["Year", "Method"]
cols_sw = ["Year", "Method"]
for crop in crops:
  cols_y.append(crop.capitalize()+"(mean)")
  cols_y.append(crop.capitalize()+"(std)")
  cols_sw.append(crop.capitalize()+"(mean)")
  cols_sw.append(crop.capitalize()+"(std)")
df_err_y = pd.DataFrame(columns=cols_y)
df_err_sw = pd.DataFrame(columns=cols_sw)

i = 0
for year in years:
  df_g = df_out[(df_out["year"]==year)&(df_out["method"]=="g")]
  df_b = df_out[(df_out["year"]==year)&(df_out["method"]=="b")]
  df_t = df_out[(df_out["year"]==year)&(df_out["method"]=="t")]

  row_y = [year, "Generative"]
  row_sw = [year, "Generative"]
  for crop in crops:
    # Yield
    true = df_t[df_t["crop"]==crop]["yield"].item()
    gen = df_g[df_g["crop"]==crop]["yield"]
    row_y.append(abs(gen-true).mean())
    row_y.append(abs(gen-true).std())
    # Soil water
    true = df_t[df_t["crop"]==crop]["water"].item()
    gen = df_g[df_g["crop"]==crop]["water"]
    row_sw.append(abs(gen-true).mean())
    row_sw.append(abs(gen-true).std())

  df_err_y.loc[i] = row_y
  df_err_sw.loc[i] = row_sw
  i += 1

  row_y = [year, "Conventional"]
  row_sw = [year, "Conventional"]
  for crop in crops:
    # Yield
    true = df_t[df_t["crop"]==crop]["yield"].item()
    bench = df_b[df_b["crop"]==crop]["yield"]
    row_y.append(abs(bench-true).mean())
    row_y.append(abs(bench-true).std())
    # Soil water
    true = df_t[df_t["crop"]==crop]["water"].item()
    bench = df_b[df_b["crop"]==crop]["water"]
    row_sw.append(abs(bench-true).mean())
    row_sw.append(abs(bench-true).std())

  df_err_y.loc[i] = row_y
  df_err_sw.loc[i] = row_sw
  i += 1
df_err_y[cols_y[2:]] = df_err_y[cols_y[2:]].astype("int")
df_err_sw = df_err_sw.round(1)


# %%
"""Write a tex file per year"""
filename = "APSIMerrors1.tex"
for c in df_err_y.columns[2:]:
  df_err_y[c] = df_err_y[c].map("{:,}".format)
df_err_y.to_latex(filename, index=False, float_format="%g")
with open(filename, "r") as f:
  lines = f.readlines()
with open(filename, "w") as f:
  for line in lines:
    if not ("rule" in line):
      line = line.replace("\\\\\n", "\\\\\\hline\n")
      f.write(line)


# %%
