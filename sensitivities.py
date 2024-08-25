import time
from helpers.helpers import *
from SoP import SoP_connectivity
from joblib import delayed, Parallel
from os import cpu_count


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


if __name__ == '__main__':
    start = time.time()

    FILE = 'Geometries/cube_cutout.stl'
    geometry = pv.read(FILE)
    geometry = geometry.subdivide(2, subfilter='linear')
    geometry = prep_mesh(geometry, decimation=0)

    # set parameters
    print('Generating connectivity')
    connectivity = read_connectivity_csv('out/sim_data/connectivity2.csv')
    print(f'Connectivity took {time.time() - start} seconds')

    args = {
        'connectivity': connectivity,
        'build_dir': np.array([0, 0, 1]),
        'down_thresh': np.sin(np.deg2rad(45)),
        'up_thresh': np.sin(np.deg2rad(0)),
        'down_k': 10,
        'up_k': 20,
        'SoP_penalty': 0,
        'softmin_p': -140
    }

    steps = np.logspace(-10, 0, 15)

    x = np.deg2rad([30, -30])
    finite_differences_plot(SoP_connectivity, x, geometry, args, steps, 'forward'
                            , 'out/supportvolume/SoP/finite_forward_differences_3030_normalized.svg')

    stop = time.time()
    print(f'Time taken: {stop - start} seconds')
