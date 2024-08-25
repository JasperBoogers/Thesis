import os.path
import time
import logging
from helpers.helpers import *
from SoP import SoP_connectivity, SoP_connectivity_no_deriv, SoP_connectivity_penalty
from sensitivities import calc_cell_sensitivities
from scipy.optimize import minimize, differential_evolution, Bounds
from scipy.stats import qmc
from matplotlib import pyplot as plt
from os import path, mkdir
from datetime import datetime as dt

logger = logging.getLogger(__name__)


def case_study_smoothing(mesh, par, k_range, savename):
    # copy par dict
    fun_arg = par.copy()

    a_max = np.deg2rad(180)
    step = 201
    dim = 'x'

    fun = []
    dfun = []
    angles = []
    for k in k_range:
        logging.debug(f'k = {k}')
        fun_arg['down_k'] = k
        angles, f, da, db = grid_search_1D(SoP_connectivity, mesh, fun_arg, a_max, step, dim)

        # add to data lists
        fun.append(f)
        if dim == 'x':
            dfun.append(da)
        else:
            dfun.append(db)

    # make & save plots
    fig1, ax1 = plt.subplots(1, 1)
    for i, k in enumerate(k_range):
        ax1.plot(np.rad2deg(angles), fun[i], label=f'k={k}')
    ax1.set_ylabel(r'Volume [mm$^3$]')
    ax1.set_xlabel(f'Rotation about {dim}-axis [deg]')
    ax1.legend()
    plt.savefig(f'{savename}_smoothing_comp.svg', format='svg', bbox_inches='tight')

    fig2, ax2 = plt.subplots(1, 1)
    for i, k in enumerate(k_range):
        ax2.plot(np.rad2deg(angles), dfun[i], label=f'k={k}')
    ax2.set_ylabel(r'Volume derivative [mm$^3$/deg]')
    ax2.set_xlabel(f'Rotation about {dim}-axis [deg]')
    ax2.legend()
    plt.savefig(f'{savename}_smoothing_comp_derivative.svg', format='svg', bbox_inches='tight')

    fig1.show()
    fig2.show()

    # write data lists to csv
    logging.debug(f'Write to csv')
    write_line_search_csv('k range', k_range, fun, dfun, f'{savename}_smoothing_comp.csv')


def case_study_penalty(mesh, par, penalty_range, savename):
    # copy par dict
    fun_arg = par.copy()

    a_max = np.deg2rad(180)
    step = 201
    dim = 'y'

    fun = []
    dfun = []
    angles = []
    for k in penalty_range:
        logging.debug(f'Penalty = {k}')
        fun_arg['SoP_penalty'] = k
        angles, f, da, db = grid_search_1D(SoP_connectivity_penalty, mesh, fun_arg, a_max, step, dim)

        # add to data lists
        fun.append(f)
        if dim == 'x':
            dfun.append(da)
        else:
            dfun.append(db)

    # make & save plots
    fig1, ax1 = plt.subplots(1, 1)
    for i, k in enumerate(penalty_range):
        ax1.plot(np.rad2deg(angles), fun[i], label=fr'b={k}')
    ax1.set_ylabel(r'Volume [mm$^3$]')
    ax1.set_xlabel(f'Rotation about {dim}-axis [deg]')
    ax1.legend()
    plt.savefig(f'{savename}_penalty_comp.svg', format='svg', bbox_inches='tight')

    fig2, ax2 = plt.subplots(1, 1)
    for i, k in enumerate(penalty_range):
        ax2.plot(np.rad2deg(angles), dfun[i], label=fr'b={k}')
    ax2.set_ylabel(r'Volume derivative [mm$^3$/deg]')
    ax2.set_xlabel(f'Rotation about {dim}-axis [deg]')
    ax2.legend()
    plt.savefig(f'{savename}_penalty_comp_derivative.svg', format='svg', bbox_inches='tight')

    fig1.show()
    fig2.show()

    # write data lists to csv
    logging.debug(f'Write to csv')
    write_line_search_csv('penalty range', penalty_range, fun, dfun, f'{savename}_penalty_comp.csv')


def case_study_overhang(mesh, par, threshold_range, savename):
    # copy par dict
    fun_arg = par.copy()

    a_max = np.deg2rad(180)
    step = 201
    dim = 'x'

    fun = []
    dfun = []
    angles = []
    for k in threshold_range:
        logging.debug(f'Threshold = {k}')
        fun_arg['down_thresh'] = np.sin(np.deg2rad(k))
        angles, f, da, db = grid_search_1D(SoP_connectivity, mesh, fun_arg, a_max, step, dim)

        # add to data lists
        fun.append(f)
        if dim == 'x':
            dfun.append(da)
        else:
            dfun.append(db)

    # make & save plots
    fig1, ax1 = plt.subplots(1, 1)
    for i, k in enumerate(threshold_range):
        ax1.plot(np.rad2deg(angles), fun[i], label=f'{k} deg')
    ax1.set_ylabel(r'Volume [mm$^3$]')
    ax1.set_xlabel(f'Rotation about {dim}-axis [deg]')
    ax1.legend()
    plt.savefig(f'{savename}_overhang_comp.svg', format='svg', bbox_inches='tight')

    fig2, ax2 = plt.subplots(1, 1)
    for i, k in enumerate(threshold_range):
        ax2.plot(np.rad2deg(angles), dfun[i], label=f'{k} deg')
    ax2.set_ylabel(r'Volume derivative [mm$^3$/deg]')
    ax2.set_xlabel(f'Rotation about {dim}-axis [deg]')
    ax2.legend()
    plt.savefig(f'{savename}_overhang_comp_derivative.svg', format='svg', bbox_inches='tight')

    fig1.show()
    fig2.show()

    # write data lists to csv
    logging.debug(f'Write to csv')
    write_line_search_csv('Overhang threshold', threshold_range, fun, dfun, f'{savename}_overhang_comp.csv')


def case_study_up_thresh(mesh, par, up_range, savename):
    # copy par dict
    fun_arg = par.copy()

    a_max = np.deg2rad(180)
    step = 201
    dim = 'x'

    fun = []
    dfun = []
    angles = []
    for k in up_range:
        logging.debug(f'Threshold = {k}')
        fun_arg['up_thresh'] = np.sin(np.deg2rad(k))
        angles, f, da, db = grid_search_1D(SoP_connectivity, mesh, fun_arg, a_max, step, dim)

        # add to data lists
        fun.append(f)
        if dim == 'x':
            dfun.append(da)
        else:
            dfun.append(db)

        # make & save plots
    fig1, ax1 = plt.subplots(1, 1)
    for i, k in enumerate(up_range):
        ax1.plot(np.rad2deg(angles), fun[i], label=f'{k} deg')
    ax1.set_ylabel(r'Volume [mm$^3$]')
    ax1.set_xlabel(f'Rotation about {dim}-axis [deg]')
    ax1.legend()
    plt.savefig(f'{savename}_upward_comp.svg', format='svg', bbox_inches='tight')

    fig2, ax2 = plt.subplots(1, 1)
    for i, k in enumerate(up_range):
        ax2.plot(np.rad2deg(angles), dfun[i], label=f'{k} deg')
    ax2.set_ylabel(r'Volume derivative [mm$^3$/deg]')
    ax2.set_xlabel(f'Rotation about {dim}-axis [deg]')
    ax2.legend()
    plt.savefig(f'{savename}_upward_comp_derivative.svg', format='svg', bbox_inches='tight')

    fig1.show()
    fig2.show()

    # write data lists to csv
    logging.debug(f'Write to csv')
    write_line_search_csv('Upward threshold', up_range, fun, dfun, f'{savename}_up_thresh_comp.csv')


def case_study_optimizers(mesh, par, methods, savename):
    def callbck(p):
        # local opt_x
        opt_x.append(p)

    # copy par dict
    fun_arg = par.copy()

    # grid search first
    logging.debug('Perform grid search')
    steps = 21
    max_angle = np.deg2rad(180)
    grid_start = time.time()
    ax, ay, f = grid_search_no_deriv(SoP_connectivity_no_deriv, mesh, fun_arg, max_angle, steps)
    grid_end = time.time()
    logging.debug(f'Grid search finished in {grid_end - grid_start} seconds')

    # save
    logging.debug('Save grid search to csv')
    write_csv(f, f'{savename}_grid_search.csv')

    # make contour plot
    logging.debug('Save contour plot')
    make_contour_plot(np.rad2deg(ax), np.rad2deg(ay), f, None, f'{savename}_Contour_plot.svg')

    # setup results df
    grad_methods, fd_methods = methods
    res = pd.DataFrame(
        columns=['Method', 'Function value', 'x min', '# evaluations', 'Duration', 'Jacobian', 'x steps'])

    # compare optimization results for each method
    x0 = np.deg2rad([-90, -70])
    logging.debug(f'x0={x0}')
    opt_x = []
    for m in grad_methods:
        logging.info(f'Performing gradient based run for method {m}')

        del opt_x
        opt_x = [x0]
        opt_start = time.time()
        y = minimize(SoP_connectivity, x0, jac=True, args=(mesh, fun_arg), method=m, tol=1e-3, callback=callbck)
        opt_end = time.time()
        logging.info(f'Finished in {opt_end - opt_start}')
        logging.debug(y)

        d = {
            'Method': m,
            'Function value': y.fun,
            'x min': np.rad2deg(y.x),
            '# evaluations': y.nfev,
            'Duration': opt_end - opt_start,
            'Jacobian': y.jac,
            'x steps': np.rad2deg(opt_x)
        }
        res = pd.concat([res, pd.DataFrame([d])], ignore_index=True)

    for m in fd_methods:
        logging.info(f'Performing finite differences based for method {m}')

        del opt_x
        opt_x = [x0]
        opt_start = time.time()
        y = minimize(SoP_connectivity_no_deriv, x0, jac='2-point', args=(mesh, fun_arg), method=m, tol=1e-3,
                     callback=callbck)
        opt_end = time.time()
        logging.info(f'Finished in {opt_end - opt_start}')
        logging.debug(y)

        # append results
        try:
            jac = y.jac
        except AttributeError:
            jac = np.nan

        d = {
            'Method': f'FD: {m}',
            'Function value': y.fun,
            'x min': np.rad2deg(y.x),
            '# evaluations': y.nfev,
            'Duration': opt_end - opt_start,
            'Jacobian': jac,
            'x steps': np.rad2deg(opt_x)
        }
        res = pd.concat([res, pd.DataFrame([d])], ignore_index=True)

    # make contour plot with steps visualized
    xx, yy = np.meshgrid(np.rad2deg(ax), np.rad2deg(ay))
    _ = plt.figure()
    cp = plt.contour(xx, yy, f)
    for idx, row in res.iterrows():
        data = (row['x steps'] + 180) % (2 * 180) - 180
        plt.plot(data[:, 0], data[:, 1], '-o', markersize=4, label=row['Method'])
    plt.xlabel('Rotation about x-axis [deg]')
    plt.ylabel('Rotation about y-axis [deg]')
    plt.legend()
    plt.savefig(f'{savename}_opt_steps.svg', format='svg', bbox_inches='tight')

    plt.clabel(cp, inline=True, fontsize=8)
    plt.savefig(f'{savename}_opt_steps_label.svg', format='svg', bbox_inches='tight')
    plt.show()

    # save results to csv
    res.to_excel(f'{savename}_opt_comp.xlsx')

    # make gif
    # opt_steps_gif(mesh, fun_arg, res.loc[res['Method'] == 'BFGS', 'x steps'].values, savename)


def case_study_GA(mesh, par, savename):
    def callbck(p):
        # local opt_x
        opt_x.append(p)

    # copy par dict
    fun_arg = par.copy()

    # load contour plot
    max_angle = 180
    steps = 21
    if path.isfile(f'{savename}_grid_search.csv'):
        ax = np.linspace(-max_angle, max_angle, steps)
        ay = np.linspace(-max_angle, max_angle, steps)
        contour = read_csv(f'{savename}_grid_search.csv')
    else:
        ax, ay, contour = grid_search_no_deriv(SoP_connectivity_no_deriv, mesh, fun_arg, np.deg2rad(max_angle), steps)

    # opt parameters
    bounds = Bounds([-np.pi, -np.pi], [np.pi, np.pi])
    NUM_START = 4
    sampler = qmc.LatinHypercube(d=2)
    starts = qmc.scale(sampler.random(n=NUM_START), [-np.pi, -np.pi], [np.pi, np.pi])
    steps = []
    x_min = []
    n_fev = 0
    func = []
    jac = []

    opt_start = time.time()
    for n in range(NUM_START):
        logging.info(f'Iteration #{n + 1} of {NUM_START} using BFGS')
        x0 = starts[n]
        opt_x = [x0]

        logging.debug(f'x0={np.rad2deg(x0)}')
        y = minimize(SoP_connectivity, x0, jac=True, args=(mesh, fun_arg), method='BFGS', tol=1e-3, callback=callbck)

        # append results
        steps.append(opt_x)
        n_fev += y.nfev
        func.append(y.fun)
        x_min.append(y.x)
        jac.append(y.jac)

        logging.debug(y)

    logging.debug(f'Finished in {time.time() - opt_start}')
    logging.info(f'Start of GA')
    y = differential_evolution(SoP_connectivity_no_deriv, bounds, args=(mesh, fun_arg), workers=-1)
    logging.debug(f'Finished in {time.time() - opt_start}')
    logging.debug(y)
    logging.debug(f'# of evaluations: {y.nfev}')
    logging.debug(f'Population shape: {y.population.shape}')

    idx = np.argmin(func)
    logging.debug(f'Gradient opt stats:')
    logging.debug(f'Optimum: {func[idx]} at {np.rad2deg(x_min[idx])} degrees')
    logging.debug(f'Jacobian: {jac[idx]}')
    logging.debug(f'Total # of evaluations: {n_fev}')

    # check if ax and ay are in degrees
    if max(ax) <= np.pi:
        ax = np.rad2deg(ax)
        ay = np.rad2deg(ay)

    # plot steps in contour plot
    xx, yy = np.meshgrid(ax, ay)
    _ = plt.figure()
    _ = plt.contour(xx, yy, contour)
    for idx, row in enumerate(steps):
        data = (np.rad2deg(row) + 180) % (2 * 180) - 180
        plt.plot(data[:, 0], data[:, 1], '-o', markersize=4, label=idx)
    plt.xlabel('Rotation about x-axis [deg]')
    plt.ylabel('Rotation about y-axis [deg]')
    plt.legend()
    plt.savefig(f'{savename}_opt_comp_steps.svg', format='svg', bbox_inches='tight')
    plt.show()


def opt_steps_gif(mesh, args, x, filename):
    mesh_rot = calc_cell_sensitivities(mesh, np.deg2rad(x[0]), args)
    d = calc_min_projection_distance(mesh_rot)
    cam_offset = 10

    p = pv.Plotter(off_screen=True, notebook=False)
    p.add_mesh(mesh_rot, name='mesh', lighting=True, scalars='MA',
               scalar_bar_args={"title": "Support requirement"}, clim=[-1, 1], cmap='RdYlBu')
    plane = pv.Plane(center=(0, 0, mesh_rot.bounds[-2]),
                     i_size=1.5 * d,
                     j_size=1.5 * d,
                     direction=(0, 0, 1))
    p.add_mesh(plane, style='wireframe', color='k', lighting=True, name='bed')
    p.add_axes()
    p.camera.position = (15, 9, plane.bounds[-1] + cam_offset)
    p.show(interactive_update=True)

    p.open_gif(f'{filename}_opt_steps.gif', fps=1)

    for xi in x[1:]:
        mesh_rot = calc_cell_sensitivities(mesh, np.deg2rad(xi), args)

        _ = p.add_mesh(mesh_rot, name='mesh', lighting=True, scalars='MA',
                       scalar_bar_args={"title": "Support requirement"}, clim=[-1, 1], cmap='RdYlBu')
        plane = pv.Plane(center=(0, 0, mesh_rot.bounds[-2]),
                         i_size=2.2 * d,
                         j_size=2.2 * d,
                         direction=(0, 0, 1))
        p.add_mesh(plane, style='wireframe', color='k', lighting=True, name='bed')
        p.camera.position = (15, 9, plane.bounds[-1] + cam_offset)
        p.update()
        p.write_frame()

    p.close()


def plot_opt_steps(mesh, args, x):
    for step in x:
        p = pv.Plotter()
        mesh_rot = calc_cell_sensitivities(mesh, np.deg2rad(step), args)
        p.add_mesh(mesh_rot, name='mesh', scalars='V', scalar_bar_args={"title": "Support volume"}, clim=[-1, 1],
                   cmap='RdYlBu', show_edges=True, lighting=False)
        p.add_axes()
        p.camera.position = (-10, -35, 10)
        p.show()


def case_study_time(geometry, par, sizes, savename, cube=False):
    fun_arg = par.copy()

    conn_times = []
    rot_times = []
    volumes = []
    n_cell = []

    for i in sizes:

        # reduce mesh size, if considering a cube use subdivide instead of decimation
        if cube:
            mesh_div = geometry.subdivide(i, 'linear')
            mesh_div = prep_mesh(mesh_div, decimation=0)
        else:
            mesh_div = decimate_quadric(geometry, i)
            mesh_div = prep_mesh(mesh_div, scaling=0.05)
        n_cell.append(mesh_div.n_cells)

        logging.info(f'Mesh has size {mesh_div.n_cells}')

        # generate connectivity
        start = time.time()
        connectivity = generate_connectivity_vtk(mesh_div)
        conn_times.append(time.time() - start)

        # no point in doing calculations if something went wrong in connectivity generation
        if len(connectivity) != mesh_div.n_cells:
            logging.warning('Number of cells is not equal to connectivity size')
            logging.warning(f'N cells: {mesh_div.n_cells}, Conn: {len(connectivity)}')
            continue

        # perform 1D grid search
        if cube:
            angles = np.deg2rad([45, 45])
        else:
            angles = np.rad2deg([0, 0])
        fun_arg['connectivity'] = connectivity
        start = time.time()
        V, _ = SoP_connectivity(angles, mesh_div, fun_arg)
        rot_times.append(time.time() - start)
        volumes.append(V)

        logging.info(f'Took {conn_times[-1] + rot_times[-1]} seconds')

    # save data
    write_csv([n_cell, conn_times, rot_times, volumes], f'{savename}_time_scaling.csv')

    # plot time to generate connectivity
    plt.figure()
    plt.loglog(n_cell, conn_times)
    plt.xlabel('Number of facets [-]')
    plt.ylabel('Time [s]')
    plt.savefig(f'{savename}_conn_time.svg', format='svg', bbox_inches='tight')
    plt.show()

    # plot time to compute volume
    plt.figure()
    plt.loglog(n_cell, rot_times)
    plt.xlabel('Number of facets [-]')
    plt.ylabel('Time [s]')
    plt.savefig(f'{savename}_rot_time.svg', format='svg', bbox_inches='tight')
    plt.show()

    # plot volume
    plt.figure()
    plt.loglog(n_cell, volumes)
    plt.xlabel('Number of facets [-]')
    plt.ylabel(r'Volume [mm$^3$]')
    plt.savefig(f'{savename}_time_volume_comp.svg', format='svg', bbox_inches='tight')
    plt.show()


def case_study():
    start = time.time()

    # file parameters
    CASENAME = 'Armadillo'
    GEOM = 'Geometries/Armadillo.stl'

    # read geometry
    m = decimate_quadric(GEOM, 10000)
    # connectivity = generate_connectivity_vtk(m)
    connectivity = read_connectivity_csv('out/sim_data/connectivity_armadillo_10000.csv')
    mesh = prep_mesh(m, scaling=0.05)
    assert len(connectivity) == mesh.n_cells

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
    plt.set_loglevel('info')

    logging.info(f'Start of case study {GEOM}')

    # SET PARAMETERS HERE
    par = {
        'connectivity': connectivity,
        'build_dir': np.array([0, 0, 1]),
        'down_thresh': np.sin(np.deg2rad(0)),
        'up_thresh': np.sin(np.deg2rad(0)),
        'down_k': 10,
        'up_k': 10,
        'softmin_p': -40,
        'SoP_penalty': 0
    }
    smoothing_range = [1, 2, 5, 10, 15]
    penalty_range = [0, 0.5, 1, 2, 5]
    overhang_range = [0, 30, 45, 60]
    up_range = [0, 5, 10, 15, 30]
    opt_methods = [['BFGS', 'CG', 'Newton-CG'], ['BFGS', 'CG', 'Nelder-Mead']]
    divs = [0, 1, 2, 3, 4, 5]
    cell_sizes = np.logspace(2, 4, 6)

    logging.info(f'{"#" * 10} Processing {CASENAME} - effect of smoothing {"#" * 10}')
    case_study_smoothing(mesh, par, smoothing_range, OUTNAME)

    logging.info(f'{"#" * 10} Processing {CASENAME} - effect of SoP penalty {"#" * 10}')
    case_study_penalty(mesh, par, penalty_range, OUTNAME)

    logging.info(f'{"#" * 10} Processing {CASENAME} - effect of overhang threshold {"#" * 10}')
    case_study_overhang(mesh, par, overhang_range, OUTNAME)
    # #
    logging.info(f'{"#" * 10} Processing {CASENAME} - effect of upward threshold {"#" * 10}')
    case_study_up_thresh(mesh, par, up_range, OUTNAME)

    logging.info(f'{"#" * 10} Processing {CASENAME} - comparing time scaling {"#" * 10}')
    case_study_time(pv.read('Geometries/cube_cutout.stl'), par, divs, os.path.join(OUTDIR, 'cube'), cube=True)
    case_study_time(GEOM, par, cell_sizes, OUTNAME)

    logging.info(f'{"#" * 10} Processing {CASENAME} - comparing optimizers {"#" * 10}')
    logging.info(f'Using gradient based methods')
    case_study_optimizers(mesh, par, opt_methods, OUTNAME)
    logging.info(f'Using GA')
    case_study_GA(mesh, par, OUTNAME)

    # make gif
    logging.info(f'{"#" * 10} Processing {CASENAME} - GIF of opt steps {"#" * 10}')
    x = [[0, 0],
         [10, -5],
         [30, -20],
         [50, -30],
         [70, -40],
         [90, -45],
         [100, -50]]
    opt_steps_gif(mesh, par, x, OUTNAME)

    # plot some opt steps
    x = [[-50., -50.],
         [-65, -30],
         [-80.52814343, -0.83880375],
         [-58.07838285, 0.20406354]]
    plot_opt_steps(mesh, par, x)

    logging.info(f'Finished in {time.time() - start} seconds')


if __name__ == '__main__':
    case_study()
