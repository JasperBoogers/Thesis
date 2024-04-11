import csv
import time
import pyvista as pv
import numpy as np
import pandas as pd
from helpers import *
from SoP import SoP_top_cover
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from os import path
from datetime import datetime as dt


if __name__ == '__main__':
    start = time.time()

    # file parameters
    GEOM = 'Geometries/bunny/bunny_coarse.stl'
    connectivity = read_connectivity_csv('out/sim_data/bunny_coarse_connectivity.csv')
    OUTDIR = 'out/case_study'
    OUTNAME = path.join(OUTDIR, 'bunny_coarse')

    # global optimization parameters
    NUM_START = 4

    # load case study table
    df = pd.read_excel('CaseStudy.xlsx')
    df['x_min'] = df['x_min'].astype('object')
    df['opt time'] = df['opt time'].astype('object')

    for i in range(len(df)):
        print(f'\nProcessing Case Study {i}')
        method = df.loc[i, 'Method']
        overhang = df.loc[i, 'Down threshold']
        down_thresh = np.sin(np.deg2rad(overhang))
        up_thresh = np.sin(np.deg2rad(df.loc[i, 'Up threshold']))
        penalty = df.loc[i, 'SoP penalty']
        D_smoothing = df.loc[i, 'Down smoothing']

        # first determine whether a grid search is needed
        if method == 'gradient' or method == 'finite diff':
            grid_start = time.time()
            print(f'Performing grid search for method {method}')
            grid_end = time.time()
            df.loc[i, 'grid time'] = grid_end - grid_start

        # do optimization for each case
        if method == 'gradient':
            opt_time = []
            for n in range(NUM_START):
                opt_start = time.time()

                print(f'Performing gradient opt for {overhang} deg overhang & SoP penalty {penalty}')

                opt_end = time.time()
                opt_time.append(opt_end - opt_start)
        elif method == 'finite diff':
            opt_time = []
            for n in range(NUM_START):
                opt_start = time.time()

                print(f'Performing finite diff opt for {overhang} deg overhang & SoP penalty {penalty}')

                opt_end = time.time()
                opt_time.append(opt_end - opt_start)
        elif method == 'PSO/GA':
            opt_start = time.time()
            print(f'Performing PSO/GA opt for {overhang} deg overhang & SoP penalty {penalty}')
            opt_end = time.time()

            df.loc[i, 'grid time'] = 0
            opt_time = opt_end - opt_start
        else:
            print(f'unknown opt method {method}, skipping')
            continue

        # save contour plot if necessary
        if df.loc[i, 'plot'] == 'contour':
            print(f'save contour plot for case {i}')

        # save opt data
        df.loc[i, 'f_min'] = 1.0
        df.at[i, 'x_min'] = [1, 2]
        df.loc[i, '# eval'] = 1000
        df.at[i, 'opt time'] = opt_time
        df.loc[i, 'opt time avg'] = np.mean(opt_time)

    # compute total time and save output data
    df['total time'] = df['opt time avg'] + df['grid time']
    df.to_excel(OUTNAME + f'_{dt.now().strftime('%d%m_%H%M%S')}' + '.xlsx')

    print(f'Finished in {time.time() - start}')
