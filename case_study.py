import csv
import time
import logging
import pyvista as pv
import numpy as np
import pandas as pd
from helpers import *
from SoP import SoP_top_cover, SoP_connectivity, SoP_connectivity_no_deriv
from scipy.optimize import minimize, differential_evolution, Bounds
from matplotlib import pyplot as plt
from os import path, mkdir
from datetime import datetime as dt
logger = logging.getLogger(__name__)


def case_study_smoothing(mesh, par, k_range, savename):
    # copy par dict
    fun_arg = par.copy()

    # init fig
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    a_max = np.deg2rad(180)
    step = 201
    dim = 'x'

    fun = []
    dfun = []
    for k in k_range:
        logging.debug(f'k = {k}')
        fun_arg['down_k'] = k
        angles, f, da, _ = grid_search_1D(SoP_connectivity, mesh, fun_arg, a_max, step, dim)

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
    logging.debug(f'Write to csv')
    write_line_search_csv('k range', k_range, fun, dfun, f'{savename}_smoothing_comp.csv')


def case_study_penalty(mesh, par, penalty_range, savename):
    # copy par dict
    fun_arg = par.copy()

    # init fig
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    a_max = np.deg2rad(180)
    step = 201
    dim = 'x'

    fun = []
    dfun = []
    for k in penalty_range:
        logging.debug(f'Penalty = {k}')
        fun_arg['SoP_penalty'] = k
        angles, f, da, _ = grid_search_1D(SoP_connectivity, mesh, fun_arg, a_max, step, dim)

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
    # copy par dict
    fun_arg = par.copy()

    # init fig
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    a_max = np.deg2rad(180)
    step = 201
    dim = 'x'

    fun = []
    dfun = []
    for k in threshold_range:
        logging.debug(f'Threshold = {k}')
        fun_arg['down_thresh'] = np.sin(np.deg2rad(k))
        angles, f, da, _ = grid_search_1D(SoP_connectivity, mesh, fun_arg, a_max, step, dim)

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
    plt.savefig(f'{savename}_overhang_comp.svg', format='svg')
    fig.show()

    # write data lists to csv
    logging.debug(f'Write to csv')
    write_line_search_csv('Overhang threshold', threshold_range, fun, dfun, f'{savename}_overhang_comp.csv')


def case_study_up_thresh(mesh, par, up_range, savename):
    # copy par dict
    fun_arg = par.copy()

    # init fig
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    a_max = np.deg2rad(180)
    step = 201
    dim = 'x'

    fun = []
    dfun = []
    for k in up_range:
        logging.debug(f'Threshold = {k}')
        fun_arg['up_thresh'] = np.sin(np.deg2rad(k))
        angles, f, da, _ = grid_search_1D(SoP_connectivity, mesh, fun_arg, a_max, step, dim)

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
    plt.savefig(f'{savename}_upward_comp.svg', format='svg')
    fig.show()

    # write data lists to csv
    logging.debug(f'Write to csv')
    write_line_search_csv('Upward threshold', up_range, fun, dfun, f'{savename}_up_thresh_comp.csv')


def case_study_optimizers(mesh, par, methods, n_start, savename):
    # copy par dict
    fun_arg = par.copy()

    # grid search first
    logging.debug('Perform grid search')
    steps = 21
    max_angle = np.deg2rad(180)
    ax = np.linspace(-max_angle, max_angle, steps)
    ay = np.linspace(-max_angle, max_angle, steps)
    grid_start = time.time()
    f = Parallel(n_jobs=cpu_count())(delayed(SoP_connectivity_no_deriv)([x, y], mesh, fun_arg) for x in ax for y in ay)
    grid_end = time.time()
    logging.debug(f'Grid search finished in {grid_end-grid_start} seconds')

    # reshape f and save
    f = np.reshape(f, (len(ax), len(ay)))
    f = np.transpose(f)
    logging.debug('Save grid search to csv')
    write_csv(f, f'{savename}_grid_search.csv')

    # make contour plot
    logging.debug('Save contour plot')
    make_contour_plot(np.rad2deg(ax), np.rad2deg(ay), f, f'Support volume contour plot', f'{savename}_Contour_plot.svg')

    # setup results df
    grad_methods, fd_methods = methods
    res = pd.DataFrame(columns=['Method', 'Function value', 'x min', '# evaluations', 'Duration', 'Jacobian'])

    # compare optimization results for each method
    x0 = extract_x0(ax, ay, f, n_start)
    for m in grad_methods:
        fun = []
        xmin = []
        nfev = []
        opt_time = []
        jac = []
        for n in range(n_start):
            x = x0[n]
            logging.info(f'Performing gradient based run {n+1} for method {m}')
            logging.debug(f'Initial point: {np.rad2deg(x)}')

            opt_start = time.time()
            y = minimize(SoP_connectivity, x, jac=True, args=(mesh, fun_arg), method=m, tol=1e-3)
            opt_end = time.time()
            logging.debug(f'Finished in {opt_end - opt_start}')
            logging.debug(y)

            # append results
            fun.append(y.fun)
            xmin.append(y.x)
            nfev.append(y.nfev)
            opt_time.append(opt_end - opt_start)
            jac.append(y.jac)

        idx = np.argmin(np.array(fun))

        d = {
            'Method': m,
            'Function value': fun[idx],
            'x min': xmin[idx],
            '# evaluations': sum(nfev),
            'Duration': sum(opt_time),
            'Jacobian': jac[idx]
        }
        res = pd.concat([res, pd.DataFrame([d])], ignore_index=True)

    for m in fd_methods:
        fun = []
        xmin = []
        nfev = []
        opt_time = []
        jac = []
        for n in range(n_start):
            x = x0[n]
            logging.info(f'Performing finite differences based run {n+1} for method {m}')
            logging.debug(f'Initial point: {np.rad2deg(x)}')

            opt_start = time.time()
            y = minimize(SoP_connectivity_no_deriv, x, jac='2-point', args=(mesh, fun_arg), method=m, tol=1e-3)
            opt_end = time.time()
            logging.debug(f'Finished in {opt_end - opt_start}')
            logging.debug(y)

            # append results
            fun.append(y.fun)
            xmin.append(y.x)
            nfev.append(y.nfev)
            opt_time.append(opt_end - opt_start)
            try:
                jac.append(y.jac)
            except AttributeError:
                jac.append(np.nan)

        idx = np.argmin(np.array(fun))

        d = {
            'Method': f'FD: {m}',
            'Function value': fun[idx],
            'x min': xmin[idx],
            '# evaluations': sum(nfev),
            'Duration': sum(opt_time),
            'Jacobian': jac[idx]
        }
        res = pd.concat([res, pd.DataFrame([d])], ignore_index=True)

    # save results to csv
    res.to_excel(f'{savename}_opt_comp.xlsx')


def case_study_GA(mesh, par, savename):
    # copy par dict
    fun_arg = par.copy()

    bounds = Bounds([-np.pi, -np.pi], [np.pi, np.pi])
    opt_start = time.time()
    y = differential_evolution(SoP_connectivity_no_deriv, bounds, args=(mesh, fun_arg), workers=-1)
    opt_end = time.time()
    logging.debug(f'Finished in {opt_end - opt_start}')
    logging.debug(y)


def case_study():
    start = time.time()

    # global optimization parameters
    NUM_START = 4

    # file parameters
    INPUTFILE = 'CaseStudy.xlsx'
    CASENAME = 'Bunny_coarse'
    GEOM = 'Geometries/bunny/bunny_coarse.stl'
    connectivity = read_connectivity_csv('out/sim_data/bunny_coarse_connectivity.csv')
    # CASENAME = 'cutout_cube'
    # GEOM = 'Geometries/cube_cutout.stl'
    # connectivity = read_connectivity_csv('out/sim_data/connectivity2.csv')

    # read geometry
    mesh = pv.read(GEOM)
    # mesh = mesh.subdivide(2, subfilter='linear')
    mesh = prep_mesh(mesh, decimation=0)

    # make new folder based on current datetime
    now = dt.now().strftime('%d%m_%H%M%S')
    OUTDIR = f'out/case_study/{now}'
    mkdir(OUTDIR)
    OUTNAME = path.join(OUTDIR, CASENAME)

    # setup logger
    logfile = f'{OUTDIR}/log.txt'
    logging.basicConfig(filename=logfile, format='%(levelname)s - %(asctime)s: %(name)s - %(message)s',
                        datefmt='%H:%M:%S')
    logging.getLogger('').setLevel(logging.DEBUG)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    # load case study table
    logging.info(f'Start of case study {GEOM}')
    # logging.info(f'Load input file {INPUTFILE}')
    # df = pd.read_excel(INPUTFILE)
    # df['x_min'] = df['x_min'].astype('object')
    # df['opt time'] = df['opt time'].astype('object')
    # df['f_min'] = df['x_min'].astype('object')
    # df['# eval'] = df['# eval'].astype('object')
    # df['Jac'] = df['Jac'].astype('object')

    # set parameters
    par = {
        'connectivity': connectivity,
        'build_dir': np.array([0, 0, 1]),
        'down_thresh': np.sin(np.deg2rad(0)),
        'up_thresh': np.sin(np.deg2rad(0)),
        'down_k': 10,
        'up_k': 20,
        'plane_offset': calc_min_projection_distance(mesh),
        'SoP_penalty': 1
    }
    smoothing_range = [5, 10, 15, 20, 25]
    penalty_range = [0.8, 1, 1.1, 1.2, 1.5, 2]
    overhang_range = [0, 30, 45, 60, 90]
    up_range = [0, 5, 10, 15, 20]
    opt_methods = [['BFGS', 'CG', 'Newton-CG'], ['BFGS', 'CG', 'Nelder-Mead']]

    logging.info(f'{"#" * 10} Processing {CASENAME} - effect of smoothing {"#" * 10}')
    case_study_smoothing(mesh, par, smoothing_range, OUTNAME)

    logging.info(f'{"#" * 10} Processing {CASENAME} - effect of SoP penalty {"#" * 10}')
    case_study_penalty(mesh, par, penalty_range, OUTNAME)

    logging.info(f'{"#" * 10} Processing {CASENAME} - effect of overhang threshold {"#" * 10}')
    case_study_overhang(mesh, par, overhang_range, OUTNAME)

    logging.info(f'{"#" * 10} Processing {CASENAME} - effect of upward threshold {"#" * 10}')
    case_study_up_thresh(mesh, par, up_range, OUTNAME)

    logging.info(f'{"#" * 10} Processing {CASENAME} - comparing optimizers {"#" * 10}')
    logging.info(f'Using gradient based methods')
    case_study_optimizers(mesh, par, opt_methods, NUM_START, OUTNAME)
    logging.info(f'Using GA')
    case_study_GA(mesh, par, OUTNAME)

    logging.info(f'Finished in {time.time() - start} seconds')


if __name__ == '__main__':
    case_study()
