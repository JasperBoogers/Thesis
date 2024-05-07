import time
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from helpers import *
from io_helpers import read_connectivity_csv
from support_volume_3D import support_volume_smooth, support_volume_analytic
from support_volume_2D import support_2D
from SoP import SoP_connectivity
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

    diffx = abs((np.array(fx) - dfdx) / dfdx)
    diffy = abs((np.array(fy) - dfdy) / dfdy)

    _ = plt.figure()
    _ = plt.loglog(h_range, diffx, 'b', label=r'$V_{,\alpha}$')
    _ = plt.loglog(h_range, diffy, 'k', label=r'$V_{,\beta}$')
    # plt.title(title)
    plt.xlabel('Step size [-]')
    # plt.ylabel(r'$\frac{|\tilde{V}_{,\theta} - V_{,\theta}|}{|V_{,\theta}|}$')
    plt.ylabel('Relative error [-]')
    _ = plt.legend()
    if outfile is not None:
        plt.savefig(outfile, format='svg', bbox_inches='tight')
    plt.show()


def calc_cell_sensitivities(mesh: pv.PolyData | pv.DataSet, angles: list | np.ndarray, par):
    p = par['softmin_p']

    _, _, R, dRda, dRdb = construct_rotation_matrix(angles[0], angles[1])
    mesh_rot = rotate_mesh(mesh, R)

    # set z_min
    z_min, dz_min = mellow_min(mesh_rot.points, p)
    dzda = np.sum(dz_min * np.transpose(dRda @ np.transpose(mesh.points)), axis=0)
    dzdb = np.sum(dz_min * np.transpose(dRdb @ np.transpose(mesh.points)), axis=0)

    # compute average coordinate for each cell, and store in 'Center' array
    mesh_rot.cell_data['Center'] = [np.sum(c.points, axis=0) / 3 for c in mesh_rot.cell]

    M, dMda, dMdb = smooth_overhang_connectivity(mesh, mesh_rot, R, dRda, dRdb, par)

    # compute sensitivities for all cells
    A, dAda, dAdb, h, dhda, dhdb = calc_V_vectorized(mesh, mesh_rot, dRda, dRdb, z_min, dzda, dzdb, par)

    mesh_rot.cell_data['MA'] = M * A / mesh_rot.cell_data['Area']
    mesh_rot.cell_data['M'] = M / mesh_rot.cell_data['Area']
    mesh_rot.cell_data['V'] = (M * A * h) / mesh_rot.cell_data['Area']
    mesh_rot.cell_data['Volume'] = (M * A * h)
    mesh_rot.cell_data['dVda'] = (M * A * dhda + M * dAda * h + dMda * A * h) / mesh_rot.cell_data['Area']
    mesh_rot.cell_data['dVdb'] = (M * A * dhdb + M * dAdb * h + dMdb * A * h) / mesh_rot.cell_data['Area']
    return mesh_rot


def plot_cell_sensitivities(mesh: pv.PolyData | pv.DataSet, axis: str = 'x') -> None:
    p = pv.Plotter()
    if axis == 'x':
        scalars = 'dVda'
    elif axis == 'y':
        scalars = 'dVdb'
    elif axis == 'V':
        scalars = 'V'
    else:
        scalars = 'MA'

    _ = p.add_mesh(mesh, lighting=False, scalars=scalars, show_edges=True)
    p.add_axes()
    p.show()


def func(angles, m, arg):
    a, b = angles
    v = a ** 2 * b ** 2
    dvda = 2 * a * b ** 2
    dvdb = 2 * a ** 2 * b
    return v, [dvda, dvdb]


def func32(angles, m, arg):
    a, b = angles
    v = a ** 2 * b ** 2
    dvda = 2 * a * b ** 2
    dvdb = 2 * a ** 2 * b
    return np.float32(v), np.float32([dvda, dvdb])


if __name__ == '__main__':
    start = time.time()
    #
    # load file
    # FILE = 'Geometries/bunny/bunny_coarse.stl'
    FILE = 'Geometries/jet_bracket_2.stl'
    m = pv.read(FILE)
    # m = m.subdivide(2, subfilter='linear')
    m = prep_mesh(m, decimation=0.8)
    connectivity = read_connectivity_csv('out/sim_data/jet_bracket_2_conn.csv')

    assert len(connectivity) == m.n_cells
    args = {
        'connectivity': connectivity,
        'build_dir': np.array([0, 0, 1]),
        'down_thresh': np.sin(np.deg2rad(45)),
        'up_thresh': np.sin(np.deg2rad(0)),
        'down_k': 10,
        'up_k': 10,
        'SoP_penalty': 1,
        'softmin_p': -15
    }

    m = calc_cell_sensitivities(m, np.deg2rad([481.92160597, 179.79582416]), args)
    m.plot(scalars='V', scalar_bar_args={"title": "Support volume"}, cmap='RdYlBu', clim=[-1, 1],
           show_edges=True, lighting=False)
    steps = np.logspace(-10, 0, 10)

    x = np.deg2rad([30, -30])
    finite_differences_plot(SoP_connectivity, x, m, args, steps, 'forward'
                            , 'out/supportvolume/SoP/finite_forward_differences_5010_normalized.svg')

    # FILE = 'Geometries/cube.stl'
    # m = pv.read(FILE)
    # m = prep_mesh(m, decimation=0)
    # m = m.subdivide(2, subfilter='linear')
    #
    # # # set parameters
    # OVERHANG_THRESHOLD = -1e-5
    # PLANE_OFFSET = calc_min_projection_distance(m)
    # args = [OVERHANG_THRESHOLD, PLANE_OFFSET]
    #
    # # x = np.deg2rad([10, 10])
    # x = [1, 1]
    # #
    # steps = np.logspace(-10, 0, 10)
    #
    # fx = []
    # fx2 = []
    # # f, dfdx, _ = support_2D(x, points, faces, normals, plane)
    # f, [dfdx, _] = func32(x, m, args)
    # f2, [dfdx2, _] = func(x, m, args)
    # for h in steps:
    #     v, w = x
    #     fhx, [_, _] = func32([v+h, w], m, args)
    #     fhx2, [_, _] = func([v+h, w], m, args)
    #     # fhx, _, _ = support_2D([x[0] + h], points, faces, normals, plane)
    #     dx = (fhx - f)/h
    #     dx2 = (fhx2 - f2)/h
    #
    #     fx.append((dx - dfdx)/dfdx)
    #     fx2.append((dx2 - dfdx2)/dfdx2)
    #
    # plt.loglog(steps, abs(np.array(fx2)), label='double precision')
    # plt.loglog(steps, abs(np.array(fx)), label='single precision')
    # plt.xlabel('Step size [-]')
    # plt.ylabel('Relative error [-]')
    # plt.legend()
    # plt.title(r'Relative error for $f=x^{2}y^{2}$ at [1, 1]')
    # plt.savefig('out/precision_error.svg', format='svg', bbox_inches='tight')
    # plt.show()
    # stop = time.time()
    # print(f'Time taken: {stop - start} seconds')
