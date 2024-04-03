import time
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from helpers import *
from SoP_threshold import SoP_smooth
from joblib import delayed, Parallel


def fwd(fun, angles, mesh, threshold, plane, h):
    fplush_x, _ = fun([angles[0] + h, angles[1]], mesh, threshold, plane)
    fplush_y, _ = fun([angles[0], angles[1] + h], mesh, threshold, plane)

    return fplush_x, fplush_y


def cntrl2(fun, angles, mesh, threshold, plane, h):
    x1, _ = fun([angles[0] + 2 * h, angles[1]], mesh, threshold, plane)
    x2, _ = fun([angles[0] + h, angles[1]], mesh, threshold, plane)
    x3, _ = fun([angles[0] - h, angles[1]], mesh, threshold, plane)
    x4, _ = fun([angles[0] - 2 * h, angles[1]], mesh, threshold, plane)

    y1, _ = fun([angles[0], angles[1] + 2 * h], mesh, threshold, plane)
    y2, _ = fun([angles[0], angles[1] + h], mesh, threshold, plane)
    y3, _ = fun([angles[0], angles[1] - h], mesh, threshold, plane)
    y4, _ = fun([angles[0], angles[1] - 2 * h], mesh, threshold, plane)

    x = (-x1 + 8 * x2 - 8 * x3 + x4) / (12 * h)
    y = (-y1 + 8 * y2 - 8 * y3 + y4) / (12 * h)
    return x, y


def cntrl(fun, angles, mesh, threshold, plane, h):
    x2, _ = fun([angles[0] + h, angles[1]], mesh, threshold, plane)
    x3, _ = fun([angles[0] - h, angles[1]], mesh, threshold, plane)

    y2, _ = fun([angles[0], angles[1] + h], mesh, threshold, plane)
    y3, _ = fun([angles[0], angles[1] - h], mesh, threshold, plane)

    x = (x2 - x3) / (2 * h)
    y = (y2 - y3) / (2 * h)
    return x, y


def finite_differences_plot(fun, angles, mesh, threshold, plane, h_range, method='forward', outfile=None):
    # fx = []
    # fy = []
    # for h in h_range:
    #     fplush_x, _ = fun([angles[0] + h, angles[1]], mesh, threshold, plane)
    #     fplush_y, _ = fun([angles[0], angles[1] + h], mesh, threshold, plane)
    #     fx.append(fplush_x)
    #     fy.append(fplush_y)
    #
    # fx = np.array(fx)
    # fy = np.array(fy)

    if method == 'forward':
        res = Parallel(n_jobs=cpu_count())(delayed(fwd)(fun, angles, mesh, threshold, plane, h) for h in h_range)
        title = f'Forward differences at x={np.rad2deg(angles)} degrees'
    elif method == 'central':
        res = Parallel(n_jobs=cpu_count())(delayed(cntrl)(fun, angles, mesh, threshold, plane, h) for h in h_range)
        title = f'Central differences at x={np.rad2deg(angles)} degrees'
    elif method == 'central2':
        res = Parallel(n_jobs=cpu_count())(delayed(cntrl2)(fun, angles, mesh, threshold, plane, h) for h in h_range)
        title = f'Second order central differences at x={np.rad2deg(angles)} degrees'
    else:
        res = Parallel(n_jobs=cpu_count())(delayed(fwd)(fun, angles, mesh, threshold, plane, h) for h in h_range)
        title = f'Forward differences at x={np.rad2deg(angles)} degrees'

    fx, fy = zip(*res)
    f, [dfdx, dfdy] = fun(angles, mesh, threshold, plane)

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


if __name__ == '__main__':
    # load file
    FILE = 'Geometries/cube_cutout.stl'
    m = pv.read(FILE)
    m = prep_mesh(m, decimation=0)
    m = m.subdivide(2, subfilter='linear')

    # set parameters
    OVERHANG_THRESHOLD = -1e-5
    PLANE_OFFSET = calc_min_projection_distance(m)
    h_range = np.logspace(-10, 0, 50)

    start = time.time()
    x = np.deg2rad([-70, -70])
    finite_differences_plot(SoP_smooth, x, m, OVERHANG_THRESHOLD, PLANE_OFFSET, h_range, 'forward'
                            , 'out/supportvolume/SoP/finite_central_differences_7070_normalized.svg')
    stop = time.time()
    print(f'Time taken: {stop - start} seconds')
