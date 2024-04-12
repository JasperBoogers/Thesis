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


def case_study_smoothing(mesh, par, k_range, savename):

    # init fig
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    a_max = np.deg2rad(180)
    step = 201
    dim = 'x'

    fun = []
    dfun = []
    for k in k_range:
        logging.debug(f'k = {k}')
        par['down_k'] = k
        angles, f, da, _ = grid_search_1D(SoP_connectivity, mesh, par, a_max, step, dim)

        # add to plot
        ax1.plot(np.rad2deg(angles), f, label=f'k={k}')
        ax2.plot(np.rad2deg(angles), da, label=f'k={k}')

        # add to data lists
        fun.append(f)
        dfun.append(da)

    # make & save plot
    ax1.set_ylabel(r'Volume [mm$^3$]')
    ax2.set_ylabel(r'Volume derivative [mm$^3$/deg]')
    ax2.set_xlabel(f'Rotation about {dim} axis [deg]')
    ax1.legend()
    fig.suptitle(f'Effect of smoothing on support volume function')
    plt.savefig(f'{savename}_smoothing_comp.svg', format='svg')
    fig.show()

    # write data lists to csv
    write_line_search_csv('k range', k_range, fun, dfun, f'{savename}_smoothing_comp.csv')


def case_study_penalty(mesh, par, penalty_range, savename):
    # init fig
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    a_max = np.deg2rad(180)
    step = 201
    dim = 'x'

    fun = []
    dfun = []
    for k in penalty_range:
        logging.debug(f'Penalty = {k}')
        par['SoP_penalty'] = k
        angles, f, da, _ = grid_search_1D(SoP_connectivity, mesh, par, a_max, step, dim)

        # add to plot
        ax1.plot(np.rad2deg(angles), f, label=fr'$\gamma$={k}')
        ax2.plot(np.rad2deg(angles), da, label=fr'$\gamma$={k}')

        # add to data lists
        fun.append(f)
        dfun.append(da)

    # make & save plot
    ax1.set_ylabel(r'Volume [mm$^3$]')
    ax2.set_ylabel(r'Volume derivative [mm$^3$/deg]')
    ax2.set_xlabel(f'Rotation about {dim} axis [deg]')
    ax1.legend()
    fig.suptitle(f'Effect of support on part penalization on support volume function')
    plt.savefig(f'{savename}_penalty_comp.svg', format='svg')
    fig.show()

    # write data lists to csv
    logging.debug(f'Write to csv')
    write_line_search_csv('penalty range', penalty_range, fun, dfun, f'{savename}_penalty_comp.csv')


def case_study_overhang(mesh, par, threshold_range, savename):
    # init fig
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    a_max = np.deg2rad(180)
    step = 201
    dim = 'x'

    fun = []
    dfun = []
    for k in threshold_range:
        logging.debug(f'Threshold = {k}')
        par['down_thresh'] = np.sin(np.deg2rad(k))
        angles, f, da, _ = grid_search_1D(SoP_connectivity, mesh, par, a_max, step, dim)

        # add to plot
        ax1.plot(np.rad2deg(angles), f, label=fr'{k} deg')
        ax2.plot(np.rad2deg(angles), da, label=fr'{k} deg')

        # add to data lists
        fun.append(f)
        dfun.append(da)

    # make & save plot
    ax1.set_ylabel(r'Volume [mm$^3$]')
    ax2.set_ylabel(r'Volume derivative [mm$^3$/deg]')
    ax2.set_xlabel(f'Rotation about {dim} axis [deg]')
    ax1.legend()
    fig.suptitle(f'Effect of overhang threshold on support volume function')
    plt.savefig(f'{savename}_penalty_comp.svg', format='svg')
    fig.show()

    # write data lists to csv
    logging.debug(f'Write to csv')
    write_line_search_csv('Overhang threshold', threshold_range, fun, dfun, f'{savename}_overhang_comp.csv')


def case_study_up_thresh(mesh, par, up_range, savename):
    # init fig
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    a_max = np.deg2rad(180)
    step = 201
    dim = 'x'

    fun = []
    dfun = []
    for k in up_range:
        logging.debug(f'Threshold = {k}')
        par['up_thresh'] = np.sin(np.deg2rad(k))
        angles, f, da, _ = grid_search_1D(SoP_connectivity, mesh, par, a_max, step, dim)

        # add to plot
        ax1.plot(np.rad2deg(angles), f, label=fr'{k} deg')
        ax2.plot(np.rad2deg(angles), da, label=fr'{k} deg')

        # add to data lists
        fun.append(f)
        dfun.append(da)

    # make & save plot
    ax1.set_ylabel(r'Volume [mm$^3$]')
    ax2.set_ylabel(r'Volume derivative [mm$^3$/deg]')
    ax2.set_xlabel(f'Rotation about {dim} axis [deg]')
    ax1.legend()
    fig.suptitle(f'Effect of upward face threshold on support volume function')
    plt.savefig(f'{savename}_penalty_comp.svg', format='svg')
    fig.show()

    # write data lists to csv
    logging.debug(f'Write to csv')
    write_line_search_csv('Overhang threshold', up_range, fun, dfun, f'{savename}_up_thresh_comp.csv')


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

    # set parameters
    par = {
        'connectivity': connectivity,
        'build_dir': np.array([0, 0, 1]),
        'down_thresh': np.sin(np.deg2rad(0)),
        'up_thresh': np.sin(np.deg2rad(0)),
        'down_k': 10,
        'up_k': 10,
        'plane_offset': calc_min_projection_distance(mesh),
        'SoP_penalty': 1
    }
    smoothing_range = [5, 10, 15, 20, 25]
    penalty_range = [0.8, 1, 1.1, 1.2, 1.5, 2]
    overhang_range = [0, 30, 45, 60, 90]
    up_range = [0, 5, 10, 15, 20]

    logging.info(f'{"#" * 10} Processing {CASENAME} - effect of smoothing {"#" * 10}')
    case_study_smoothing(mesh, par, smoothing_range, OUTNAME)

    logging.info(f'{"#" * 10} Processing {CASENAME} - effect of SoP penalty {"#" * 10}')
    case_study_penalty(mesh, par, penalty_range, OUTNAME)

    logging.info(f'{"#" * 10} Processing {CASENAME} - effect of overhang threshold {"#" * 10}')
    case_study_overhang(mesh, par, overhang_range, OUTNAME)

    # TODO check whether upward threshold moves upward field in postitive direction!
    logging.info(f'{"#" * 10} Processing {CASENAME} - effect of upward threshold {"#" * 10}')
    case_study_up_thresh(mesh, par, up_range, OUTNAME)

    # for i in range(len(df)):
    #     logging.info(f'{"#"*10} Processing {CASENAME} case {i} {"#"*10}')
    #     method = df.loc[i, 'Method']
    #     overhang = df.loc[i, 'Down threshold']
    #     penalty = df.loc[i, 'SoP penalty']
    #     opt_method = df.loc[i, 'Optimizer']
    #
    #
    #
    #     # first determine whether a grid search is needed
    #     if method == 'gradient' or method == 'finite diff':
    #         logging.info(f'Performing grid search for method {method}')
    #
    #         steps = 21
    #         max_angle = np.deg2rad(180)
    #
    #         ax = np.linspace(-max_angle, max_angle, steps)
    #         ay = np.linspace(-max_angle, max_angle, steps)
    #
    #         grid_start = time.time()
    #         f = Parallel(n_jobs=cpu_count())(delayed(SoP_connectivity_no_deriv)([x, y], mesh, par) for x in ax for y in ay)
    #         grid_end = time.time()
    #
    #         f = np.reshape(f, (len(ax), len(ay)))
    #         f = np.transpose(f)
    #         df.loc[i, 'grid time'] = grid_end - grid_start
    #
    #         # save contour data TODO change to write_csv
    #         with open(f'{OUTDIR}/Contour_{CASENAME}{i}.csv', 'w', newline='') as file:
    #             writer = csv.writer(file)
    #             writer.writerows(f)
    #
    #         # save contour plot if necessary
    #         if df.loc[i, 'plot'] == 'contour':
    #             logging.info(f'Save contour plot for case {i}')
    #             make_contour_plot(np.rad2deg(ax), np.rad2deg(ay), f, f'{CASENAME} {i}',
    #                               f'{OUTDIR}/Contour_{CASENAME}{i}.svg')
    #
    #         if df.loc[i, 'plot'] == 'line':
    #             logging.info(f'Save line search for case {i} in x dim')
    #             angles, f, da, db = grid_search_1D(SoP_connectivity, mesh, par, max_angle, 201, 'x')
    #
    #             with open(f'{OUTDIR}/LineSearch_{CASENAME}{i}.csv', 'w', newline='') as file:
    #                 writer = csv.writer(file)
    #                 writer.writerows(f) # TODO change to write(), f is not iterable
    #
    #             make_line_plot(np.rad2deg(angles), f, da, False, f'1D grid search - {CASENAME} {i}',
    #                            f'{OUTDIR}/LineSearch_{CASENAME}{i}.svg')
    #
    #     else:
    #         df.loc[i, 'grid time'] = 0
    #
    #     # run optimization for each case
    #     if method == 'gradient' or method == 'finite diff':
    #         opt_time = []
    #         x0 = extract_x0(ax, ay, f, NUM_START, smallest=True)
    #
    #         for n in range(NUM_START):
    #             fmin = []
    #             fev = []
    #             xmin = []
    #             jac = []
    #             x = x0[n]
    #
    #             if method == 'gradient':
    #                 logging.info(f'Performing gradient opt {n} for {overhang} deg overhang & SoP penalty {penalty}')
    #                 logging.debug(f'Initial point: {np.rad2deg(x)}')
    #
    #                 opt_start = time.time()
    #                 y = minimize(SoP_connectivity, x, jac=True, args=(mesh, par), method=opt_method)
    #                 opt_end = time.time()
    #             else:  # no gradient supplied
    #                 logging.info(f'Performing finite diff opt {n} for {overhang} deg overhang & SoP penalty {penalty}')
    #                 logging.debug(f'Initial point: {np.rad2deg(x)}')
    #
    #                 opt_start = time.time()
    #                 y = minimize(SoP_connectivity_no_deriv, x, jac='2-point', args=(mesh, par), method=opt_method)
    #                 opt_end = time.time()
    #
    #             # save data
    #             logging.info(f'Optimizer success: {y.success}')
    #             logging.debug(f'Optimal orientation found at {np.rad2deg(y.x)}')
    #             fmin.append(y.fun)
    #             jac.append(y.jac)
    #             fev.append(y.nfev)
    #             xmin.append(y.x)
    #             opt_time.append(opt_end - opt_start)
    #
    #     elif method == 'PSO/GA':
    #         logging.info(f'Performing PSO/GA opt for {overhang} deg overhang & SoP penalty {penalty}')
    #         opt_start = time.time()
    #
    #         # TODO
    #
    #         opt_end = time.time()
    #
    #         # save data
    #         logging.info(f'Optimizer success: {y.success}')
    #         logging.debug(f'Optimal orientation found at {np.rad2deg(y.x)}')
    #         fmin = y.fun
    #         jac = y.jac
    #         fev = y.nfev
    #         xmin = y.x
    #         opt_time = opt_end - opt_start
    #
    #     else:
    #         logging.warning(f'unknown opt method {method}, skipping')
    #         continue
    #
    #     # save opt data
    #     df.at[i, 'f_min'] = fmin
    #     df.at[i, 'x_min'] = xmin
    #     df.at[i, '# eval'] = fev
    #     df.at[i, 'Jac'] = jac
    #     df.at[i, 'opt time'] = opt_time
    #     df.loc[i, 'opt time avg'] = np.mean(opt_time)

    # compute total time and save output data
    # df['total time'] = df['opt time avg'] + df['grid time']
    # df.to_excel(OUTNAME + f'_{now}.xlsx')

    print(f'Finished in {time.time() - start}')


if __name__ == '__main__':
    case_study()
