"""
APSIM runs for Scenario 2

Benchmark and Truth use train.met, which is essentially the same as
the original dataset and prepared in train.py.
"""

# %%
import numpy as np
import pandas as pd
import sys, os, copy, json, subprocess
from itertools import permutations
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
Crops = [s.capitalize() for s in crops]
dir_name = "scenario2"
k = "Children" # dictionary key
met_files = glob("met files2/r*.met") # "r" for rotation
n_MC = len(met_files)

try:
  shutil.rmtree(dir_name) # Clear the directory
except:
  pass
os.makedirs(dir_name, exist_ok=True)


# %%
"""Update the template with the start and end dates"""
with open(met_files[0], "r") as f:
  lines = f.readlines()
  l = lines[24].split() # Make sure L.25 is the 1st line
  date_start = date(int(l[0]),1,1) + timedelta(int(l[1])-1)
  l = lines[-1].split()
  date_end = date(int(l[0]),1,1) + timedelta(int(l[1])-1)
with open("rotation.apsimx", "r") as f:
  data = json.load(f)
s = "%Y-%m-%dT00:00:00"
## Main
data[k][1][k][1][k][0]["Start"] = date_start.strftime(s)
data[k][1][k][1][k][0]["End"] = date_end.strftime(s)
## Factorial
data[k][1][k][0][k][0][k][2][k][0]["Start"] = date_start.strftime(s)
data[k][1][k][0][k][0][k][2][k][0]["End"] = date_end.strftime(s)
data = json.dumps(data, indent=2)
with open("rotation.apsimx", "w") as f:
  f.write(data)

month_start = date_start.month
day_start = date_start.day
month_end = date_end.month
day_end = date_end.day


# %%
"""Generated weather"""
print("Gen")
for rot in permutations(Crops):
  print(rot)
  with open("rotation.apsimx", "r") as f:
    data = json.load(f)

  x = copy.deepcopy(data[k][1])
  for _ in range(2):
    x = x[k][0]
  x = x[k]
  w = copy.deepcopy(x[0])
  r = copy.deepcopy(x[1])
  c = copy.deepcopy(x[2])
  
  # Weather
  for met in met_files:
    w[k].append(copy.deepcopy(w[k][0]))
    w[k][-1]["Name"] = met[-9:-4]
    w[k][-1]["FileName"] = "..\\met files2\\" + met[-10:]
  del w[k][0]
  
  # Rotation
  name = "".join([c[0] for c in rot]) # 3 initials
  if name != "WBC":
    r[k].append(copy.deepcopy(r[k][0]))
    r[k][-1]["Name"] = name
    arcs = copy.deepcopy(r[k][-1]["Arcs"])
    for i, Crop in enumerate(rot):
      arcs[2*i]["Conditions"][0] = "["+Crop+"Manager].Script.CanSow"
      arcs[2*i]["Actions"][0] = "["+Crop+"Manager].Script.SowCrop()"
      arcs[2*i]["DestinationName"] = Crop
      arcs[2*i+1]["Conditions"][0] = "["+Crop+"Manager].Script.CanHarvest"
      arcs[2*i+1]["Actions"][0] = "["+Crop+"Manager].Script.HarvestCrop(\"\")"
      arcs[2*i+1]["SourceName"] = Crop
    r[k][-1]["Arcs"] = arcs
    del r[k][0]

  data[k][1][k][0][k][0][k] = [w, r, c]  
  data = json.dumps(data, indent=2)
  path = dir_name + "/g" + name + ".apsimx"
  with open(path, "w") as f:
    f.write(data)

  subprocess.run([path_apsim, path])


# %%
"""Benchmark"""
print("Benchmark")
rng = np.random.default_rng(1) # Fix a random seed
for rot in permutations(Crops):
  print(rot)
  with open("rotation.apsimx", "r") as f:
    data = json.load(f)

  x = copy.deepcopy(data[k][1])
  for _ in range(2):
    x = x[k][0]
  x = x[k]
  w = copy.deepcopy(x[0])
  r = copy.deepcopy(x[1])
  c = copy.deepcopy(x[2])
  
  # Weather
  w[k][0]["FileName"] = "..\\train.met"

  # Rotation (Update the order of crops)
  name = "".join([c[0] for c in rot]) # 3 initials
  if name != "WBC":
    r[k].append(copy.deepcopy(r[k][0]))
    r[k][-1]["Name"] = name
    arcs = copy.deepcopy(r[k][-1]["Arcs"])
    for i, Crop in enumerate(rot):
      arcs[2*i]["Conditions"][0] = "["+Crop+"Manager].Script.CanSow"
      arcs[2*i]["Actions"][0] = "["+Crop+"Manager].Script.SowCrop()"
      arcs[2*i]["DestinationName"] = Crop
      arcs[2*i+1]["Conditions"][0] = "["+Crop+"Manager].Script.CanHarvest"
      arcs[2*i+1]["Actions"][0] = "["+Crop+"Manager].Script.HarvestCrop(\"\")"
      arcs[2*i+1]["SourceName"] = Crop
    r[k][-1]["Arcs"] = arcs
    del r[k][0]

  # Clock
  yrs = rng.integers(low=1989, high=2019, size=n_MC)
  for i, yr in enumerate(yrs):
    c[k].append(copy.deepcopy(c[k][0]))
    c[k][-1]["Name"] = "{:05d}".format(i)
    date_start = date(yr, month_start, day_start)
    c[k][-1]["Start"] = date_start.strftime("%Y-%m-%dT00:00:00")
    date_end = date_start + timedelta(365*3-1)
    c[k][-1]["End"] = date_end.strftime("%Y-%m-%dT00:00:00")
  del c[k][0]

  data[k][1][k][0][k][0][k] = [w, r, c]
  data = json.dumps(data, indent=2)
  path = dir_name + "/b" + name + ".apsimx"
  with open(path, "w") as f:
    f.write(data)

  subprocess.run([path_apsim, path])


# %%
"""Truth"""
for rot in permutations(Crops):
  print(rot)
  with open("rotation.apsimx", "r") as f:
    data = json.load(f)

  x = copy.deepcopy(data[k][1])
  for _ in range(2):
    x = x[k][0]
  x = x[k]
  w = copy.deepcopy(x[0])
  r = copy.deepcopy(x[1])
  c = copy.deepcopy(x[2])
  
  # Weather
  w[k][0]["FileName"] = "..\\train.met"

  # Rotation
  name = "".join([c[0] for c in rot]) # 3 initials
  if name != "WBC":
    r[k].append(copy.deepcopy(r[k][0]))
    r[k][-1]["Name"] = name
    arcs = copy.deepcopy(r[k][-1]["Arcs"])
    for i, Crop in enumerate(rot):
      arcs[2*i]["Conditions"][0] = "["+Crop+"Manager].Script.CanSow"
      arcs[2*i]["Actions"][0] = "["+Crop+"Manager].Script.SowCrop()"
      arcs[2*i]["DestinationName"] = Crop
      arcs[2*i+1]["Conditions"][0] = "["+Crop+"Manager].Script.CanHarvest"
      arcs[2*i+1]["Actions"][0] = "["+Crop+"Manager].Script.HarvestCrop(\"\")"
      arcs[2*i+1]["SourceName"] = Crop
    r[k][-1]["Arcs"] = arcs
    del r[k][0]

  data[k][1][k][0][k][0][k] = [w, r, c]  
  data = json.dumps(data, indent=2)
  path = dir_name + "/t" + name + ".apsimx"
  with open(path, "w") as f:
    f.write(data)

  subprocess.run([path_apsim, path])


# %%
"""Read results .db and put them in df_out"""
q = "SELECT `Clock.Today`, weather, clock, " +\
    "Rotation, WheatYield, BarleyYield, CanolaYield"
for i in range(1,7): # Assumed 6 soil layers
  q += ", `Soil.Water.MM({})`".format(i)
q +=  " FROM Report ORDER BY Weather, Clock"
db_files = glob(dir_name + "/*.db")
cols = ["method", "rotation", "wheat", "barley", "canola", "water"]
data = []
for db_file in db_files:
  method = db_file[-7]
  con = sqlite3.connect(db_file)
  cur = con.cursor()
  re = cur.execute(q)

  idx_w = -1
  idx_c = -1
  y_w = None
  y_b = None
  y_c = None
  sw = None
  for row in re.fetchall():
    if int(row[1]) != idx_w or int(row[2]) != idx_c:
      idx_w = int(row[1])
      idx_c = int(row[2])
    d = datetime.strptime(row[0][:10], "%Y-%m-%d").date()
    if float(row[4]) > 0:
      y_w = float(row[4])
    if float(row[5]) > 0:
      y_b = float(row[5])
    if float(row[6]) > 0:
      y_c = float(row[6])
    if d.month == month_end and d.day == day_end:
      sw = sum(row[7:13])
    if y_w is not None and\
       y_b is not None and\
       y_c is not None and\
       sw is not None:
      data.append([method, row[3], y_w, y_b, y_c, sw])
      y_w = None
      y_b = None
      y_c = None
      sw = None
df_out = pd.DataFrame(data, columns=cols)


# %%
"""Summarise df_out in terms of errors per crop/method/rotation"""
cols = ["Rotation", "Method"]
for Crop in Crops:
  cols.append(Crop+"(mean)")
  cols.append(Crop+"(std)")
cols.append("SW(mean)")
cols.append("SW(std)")
df_err = pd.DataFrame(columns=cols)

i = 0
for r in permutations(Crops):
  rot = "".join([C[0] for C in r])
  df_g = df_out[(df_out["rotation"]==rot)&(df_out["method"]=="g")]
  df_b = df_out[(df_out["rotation"]==rot)&(df_out["method"]=="b")]
  df_t = df_out[(df_out["rotation"]==rot)&(df_out["method"]=="t")]

  row = [rot, "Generative"]
  for crop in crops:
    # Yield
    true = df_t[crop].item()
    gen = df_g[crop]
    row.append(abs(gen-true).mean())
    row.append(abs(gen-true).std())
  # Soil water
  true = df_t["water"].item()
  gen = df_g["water"]
  row.append(abs(gen-true).mean())
  row.append(abs(gen-true).std())
  df_err.loc[i] = row
  i += 1

  row = [rot, "Conventional"]
  for c in crops:
    true = df_t[df_t[c]>0][c].item()
    benchmark = df_b[df_b[c]>0][c]
    row.append(abs(benchmark-true).mean())
    row.append(abs(benchmark-true).std())
  # Soil water
  true = df_t["water"].item()
  benchmark = df_b["water"]
  row.append(abs(benchmark-true).mean())
  row.append(abs(benchmark-true).std())
  df_err.loc[i] = row
  i += 1
df_err[cols[2:]] = df_err[cols[2:]].astype("int")


# %%
filename = "APSIMerrors2.tex"
df_err.to_latex(filename, index=False, float_format="%g")
with open(filename, "r") as f:
  lines = f.readlines()
with open(filename, "w") as f:
  for line in lines:
    if not ("rule" in line):
      line = line.replace("\\\\\n", "\\\\\\hline\n")
      f.write(line)


# %%
