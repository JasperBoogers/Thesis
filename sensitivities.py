import time
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from helpers import *
from io_helpers import read_connectivity_csv
from SoP_threshold import SoP_smooth
from joblib import delayed, Parallel


def fwd(fun, angles, mesh, args, h):
    fplush_x, _ = fun([angles[0] + h, angles[1]], mesh, args)
    fplush_y, _ = fun([angles[0], angles[1] + h], mesh, args)

    return fplush_x, fplush_y


def cntrl2(fun, angles, mesh, args, h):
    x1, _ = fun([angles[0] + 2 * h, angles[1]], mesh, args)
    x2, _ = fun([angles[0] + h, angles[1]], mesh, args)
    x3, _ = fun([angles[0] - h, angles[1]], mesh, args)
    x4, _ = fun([angles[0] - 2 * h, angles[1]], mesh, args)

    y1, _ = fun([angles[0], angles[1] + 2 * h], mesh, args)
    y2, _ = fun([angles[0], angles[1] + h], mesh, args)
    y3, _ = fun([angles[0], angles[1] - h], mesh, args)
    y4, _ = fun([angles[0], angles[1] - 2 * h], mesh, args)

    x = (-x1 + 8 * x2 - 8 * x3 + x4) / (12 * h)
    y = (-y1 + 8 * y2 - 8 * y3 + y4) / (12 * h)
    return x, y


def cntrl(fun, angles, mesh, args, h):
    x2, _ = fun([angles[0] + h, angles[1]], mesh, args)
    x3, _ = fun([angles[0] - h, angles[1]], mesh, args)

    y2, _ = fun([angles[0], angles[1] + h], mesh, args)
    y3, _ = fun([angles[0], angles[1] - h], mesh, args)

    x = (x2 - x3) / (2 * h)
    y = (y2 - y3) / (2 * h)
    return x, y


def finite_differences_plot(fun, angles, mesh, args, h_range, method='forward', outfile=None):

    if method == 'forward':
        res = Parallel(n_jobs=cpu_count())(delayed(fwd)(fun, angles, mesh, args, h) for h in h_range)
        title = f'Forward differences at x={np.rad2deg(angles)} degrees'
    elif method == 'central':
        res = Parallel(n_jobs=cpu_count())(delayed(cntrl)(fun, angles, mesh, args, h) for h in h_range)
        title = f'Central differences at x={np.rad2deg(angles)} degrees'
    elif method == 'central2':
        res = Parallel(n_jobs=cpu_count())(delayed(cntrl2)(fun, angles, mesh, args, h) for h in h_range)
        title = f'Second order central differences at x={np.rad2deg(angles)} degrees'
    else:
        res = Parallel(n_jobs=cpu_count())(delayed(fwd)(fun, angles, mesh, args, h) for h in h_range)
        title = f'Forward differences at x={np.rad2deg(angles)} degrees'

    fx, fy = zip(*res)
    f, [dfdx, dfdy] = fun(angles, mesh, args)

    if method == 'forward':
        fx = np.subtract(fx, f) / h_range
        fy = np.subtract(fy, f) / h_range

    diffx = abs(np.subtract(fx, dfdx) / dfdx)
    diffy = abs(np.subtract(fy, dfdy) / dfdy)

    _ = plt.figure()
    _ = plt.loglog(h_range, diffx, 'b', label=r'$V_{,\alpha}$')
    _ = plt.loglog(h_range, diffy, 'k', label=r'$V_{,\beta}$')
    plt.title(title)
    plt.xlabel('Step size [-]')
    plt.ylabel(r'$\frac{|\tilde{V}_{,\theta} - V_{,\theta}|}{|V_{,\theta}|}$')
    _ = plt.legend()
    if outfile is not None:
        plt.savefig(outfile, format='svg', bbox_inches='tight')
    plt.show()


def calc_cell_sensitivities(mesh: pv.PolyData | pv.DataSet, angles: list | np.ndarray,
                            build_dir: list | np.ndarray, z_min: list | np.ndarray):
    _, _, R, _, _ = construct_rotation_matrix(angles[0], angles[1])
    mesh = rotate_mesh(mesh, R)

    # compute cell areas
    mesh = mesh.compute_cell_sizes(length=False, volume=False)

    # compute sensitivities for all cells
    res = Parallel(n_jobs=cpu_count())(
        delayed(calc_V_under_triangle)(mesh.extract_cells(i), angles, build_dir, z_min) for i in range(mesh.n_cells))

    f, dx, dy = zip(*res)

    # f = []
    # dx = []
    # dy = []
    #
    # for i in range(mesh.n_cells):
    #     c = mesh.extract_cells(i)
    #
    #     f_, dx_, dy_ = calc_V_under_triangle(c, angles, build_dir, z_min)
    #     f.append(f_)
    #     dx.append(dx_)
    #     dy.append(dy_)

    thresh = mesh['Normals'][:, 2] < -1e-6

    mesh.cell_data['dVda'] = np.array(dx) / mesh.cell_data['Area'] * thresh
    mesh.cell_data['dVdb'] = np.array(dy) / mesh.cell_data['Area'] * thresh
    mesh.cell_data['dV'] = np.linalg.norm(np.array([dx, dy]), axis=0) / mesh.cell_data['Area'] * thresh
    mesh.cell_data['V'] = np.array(f) / mesh.cell_data['Area'] * thresh
    return mesh


def plot_cell_sensitivities(mesh: pv.PolyData | pv.DataSet, axis: str = 'x') -> None:
    p = pv.Plotter()
    if axis == 'x':
        scalars = 'dVda'
    elif axis == 'y':
        scalars = 'dVdb'
    elif axis == 'V':
        scalars = 'V'
    else:
        scalars = 'dV'

    _ = p.add_mesh(mesh, lighting=False, scalars=scalars, cmap='RdYlGn', show_edges=True)
    p.add_axes()
    p.show()


if __name__ == '__main__':
    # load file
    FILE = 'Geometries/cube_cutout.stl'
    m = pv.read(FILE)
    m = prep_mesh(m, decimation=0)
    m = m.subdivide(2, subfilter='linear')

    # set parameters
    OVERHANG_THRESHOLD = -1e-5
    PLANE_OFFSET = calc_min_projection_distance(m)
    conn = read_connectivity_csv('out/sim_data/connectivity2.csv')
    h_range = np.logspace(-10, 0, 50)

    start = time.time()
    x = np.deg2rad([45, 45])
    args = [conn, OVERHANG_THRESHOLD, PLANE_OFFSET]
    finite_differences_plot(SoP_smooth, x, m, args, h_range, 'forward'
                            , 'out/supportvolume/SoP/finite_forward_differences_4545_normalized.svg')
    stop = time.time()
    print(f'Time taken: {stop - start} seconds')
