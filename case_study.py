import csv
import time
import logging
import pyvista as pv
import numpy as np
import pandas as pd
from helpers import *
from SoP import SoP_top_cover, SoP_connectivity, SoP_connectivity_no_deriv
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from os import path, mkdir
from datetime import datetime as dt
logger = logging.getLogger(__name__)


def case_study():
    start = time.time()

    # global optimization parameters
    NUM_START = 4

    # file parameters
    INPUTFILE = 'CaseStudy.xlsx'
    CASENAME = 'Stanford Bunny'
    GEOM = 'Geometries/bunny/bunny_coarse.stl'
    mesh = pv.read(GEOM)
    # mesh = mesh.subdivide(2, subfilter='linear')
    mesh = prep_mesh(mesh, decimation=0)
    connectivity = read_connectivity_csv('out/sim_data/bunny_coarse_connectivity.csv')

    # make new folder based on current datetime
    now = dt.now().strftime('%d%m_%H%M%S')
    OUTDIR = f'out/case_study/{now}'
    mkdir(OUTDIR)
    OUTNAME = path.join(OUTDIR, 'bunny_coarse')

    # setup logger
    # logfile = path.join(OUTDIR, 'log.txt')
    logfile = f'{OUTDIR}/log.txt'
    logging.basicConfig(filename=logfile, format='%(levelname)s - %(asctime)s: %(name)s - %(message)s',
                        level=logging.INFO, datefmt='%H:%M:%S')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    # load case study table
    logging.info(f'Start of case study {GEOM} with {NUM_START}')
    logging.info(f'Load input file {INPUTFILE}')
    df = pd.read_excel(INPUTFILE)
    df['x_min'] = df['x_min'].astype('object')
    df['opt time'] = df['opt time'].astype('object')
    df['f_min'] = df['x_min'].astype('object')
    df['# eval'] = df['# eval'].astype('object')
    df['Jac'] = df['Jac'].astype('object')

    for i in range(len(df)):
        logging.info(f'{"#"*10} Processing {CASENAME} case {i} {"#"*10}')
        method = df.loc[i, 'Method']
        overhang = df.loc[i, 'Down threshold']
        penalty = df.loc[i, 'SoP penalty']
        opt_method = df.loc[i, 'Optimizer']

        # set parameters
        par = {
            'connectivity': connectivity,
            'build_dir': np.array([0, 0, 1]),
            'down_thresh': np.sin(np.deg2rad(overhang)),
            'up_thresh': np.sin(np.deg2rad(df.loc[i, 'Up threshold'])),
            'down_k': df.loc[i, 'Down smoothing'],
            'up_k': 10,
            'plane_offset': calc_min_projection_distance(mesh),
            'SoP_penalty': penalty
        }

        # first determine whether a grid search is needed
        if method == 'gradient' or method == 'finite diff':
            logging.info(f'Performing grid search for method {method}')

            steps = 21
            max_angle = np.deg2rad(180)

            ax = np.linspace(-max_angle, max_angle, steps)
            ay = np.linspace(-max_angle, max_angle, steps)

            grid_start = time.time()
            f = Parallel(n_jobs=cpu_count())(delayed(SoP_connectivity_no_deriv)([x, y], mesh, par) for x in ax for y in ay)
            grid_end = time.time()

            f = np.reshape(f, (len(ax), len(ay)))
            f = np.transpose(f)
            df.loc[i, 'grid time'] = grid_end - grid_start

            # save contour data TODO change to write_csv
            with open(f'{OUTDIR}/Contour_{CASENAME}{i}.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(f)

            # save contour plot if necessary
            if df.loc[i, 'plot'] == 'contour':
                logging.info(f'Save contour plot for case {i}')
                make_contour_plot(np.rad2deg(ax), np.rad2deg(ay), f, f'{CASENAME} {i}',
                                  f'{OUTDIR}/Contour_{CASENAME}{i}.svg')

            if df.loc[i, 'plot'] == 'line':
                logging.info(f'Save line search for case {i} in x dim')
                angles, f, da, db = grid_search_1D(SoP_connectivity, mesh, par, max_angle, 201, 'x')

                with open(f'{OUTDIR}/LineSearch_{CASENAME}{i}.csv', 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows(f) # TODO change to write(), f is not iterable

                make_line_plot(np.rad2deg(angles), f, da, False, f'1D grid search - {CASENAME} {i}',
                               f'{OUTDIR}/LineSearch_{CASENAME}{i}.svg')

        else:
            df.loc[i, 'grid time'] = 0

        # run optimization for each case
        if method == 'gradient' or method == 'finite diff':
            opt_time = []
            x0 = extract_x0(ax, ay, f, NUM_START, smallest=True)

            for n in range(NUM_START):
                fmin = []
                fev = []
                xmin = []
                jac = []
                x = x0[n]

                if method == 'gradient':
                    logging.info(f'Performing gradient opt {n} for {overhang} deg overhang & SoP penalty {penalty}')
                    logging.debug(f'Initial point: {np.rad2deg(x)}')

                    opt_start = time.time()
                    y = minimize(SoP_connectivity, x, jac=True, args=(mesh, par), method=opt_method)
                    opt_end = time.time()
                else:  # no gradient supplied
                    logging.info(f'Performing finite diff opt {n} for {overhang} deg overhang & SoP penalty {penalty}')
                    logging.debug(f'Initial point: {np.rad2deg(x)}')

                    opt_start = time.time()
                    y = minimize(SoP_connectivity_no_deriv, x, jac='2-point', args=(mesh, par), method=opt_method)
                    opt_end = time.time()

                # save data
                logging.info(f'Optimizer success: {y.success}')
                logging.debug(f'Optimal orientation found at {np.rad2deg(y.x)}')
                fmin.append(y.fun)
                jac.append(y.jac)
                fev.append(y.nfev)
                xmin.append(y.x)
                opt_time.append(opt_end - opt_start)

        elif method == 'PSO/GA':
            logging.info(f'Performing PSO/GA opt for {overhang} deg overhang & SoP penalty {penalty}')
            opt_start = time.time()

            # TODO

            opt_end = time.time()

            # save data
            logging.info(f'Optimizer success: {y.success}')
            logging.debug(f'Optimal orientation found at {np.rad2deg(y.x)}')
            fmin = y.fun
            jac = y.jac
            fev = y.nfev
            xmin = y.x
            opt_time = opt_end - opt_start

        else:
            logging.warning(f'unknown opt method {method}, skipping')
            continue

        # save opt data
        df.at[i, 'f_min'] = fmin
        df.at[i, 'x_min'] = xmin
        df.at[i, '# eval'] = fev
        df.at[i, 'Jac'] = jac
        df.at[i, 'opt time'] = opt_time
        df.loc[i, 'opt time avg'] = np.mean(opt_time)

    # compute total time and save output data
    df['total time'] = df['opt time avg'] + df['grid time']
    df.to_excel(OUTNAME + f'_{now}.xlsx')

    print(f'Finished in {time.time() - start}')


if __name__ == '__main__':
    case_study()
