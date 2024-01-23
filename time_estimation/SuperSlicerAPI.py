import os
import numpy as np
import pandas as pd
import subprocess as sp
import time
import csv
import datetime as dt
import shutil


def run_SuperSlicer(slicer, geometry, config, outfolder):
    # build command
    cmd = f'{slicer} -g {geometry} -o {outfolder} --load {config}'
    os.system(f'cmd /c {cmd}')


# define paths
CWD = os.getcwd()
SLICER_PATH = "SuperSlicer\\superslicer_console.exe"
GEOMETRY_PATH = os.path.join(CWD, "Geometries")
OUTDIR = './gcodes'
CONFIG = "config.ini"
OUTFILE = os.path.join(CWD, "time_estimation.xlsx")
DF_COLUMNS = ["File name", "Volume [mm3]", "Area [mm2]", "Height [mm]", "Height^2 [mm2]", "Height^3 [mm3]", "Time [s]"]

# get list of all geometries
geometries = os.listdir(GEOMETRY_PATH)

# check if OUTFILE exists, if not create an empty df
if os.path.isfile(OUTFILE):
    print('Found existing file ', OUTFILE)
    df = pd.read_excel(OUTFILE)
else:
    print('Creating new file ', OUTFILE)
    df = pd.DataFrame(columns=DF_COLUMNS)

# now check if new geometries have been added to Geometries dir
existing_geometries = list(df['File name'])
new_geometries = [file for file in geometries if file not in existing_geometries]

# empty the out folder
print(f'Cleaning output folder')
for file in os.listdir(OUTDIR):
    os.remove(os.path.join(OUTDIR, file))

print(f'Generating G-code for {len(new_geometries)} files')

start = time.time()
for file in geometries:
    fn = f'./Geometries/{file}'
    run_SuperSlicer(SLICER_PATH, fn, CONFIG, OUTDIR)
end = time.time()
print(f'G-code generation took {end - start} seconds')

print('finished')