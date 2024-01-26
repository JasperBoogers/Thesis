import numpy as np
import pyvista as pv
import pymeshfix as mf
from time import time
from scipy.optimize import minimize, check_grad
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from pyvista_functions import *


def support_3D_pyvista(angles: list, msh: pv.PolyData, thresh: float, plane_offset=1.0) -> float:
    # rotate
    R = Rotation.from_euler('xyz', np.append(angles, 0))
    msh = rotate_mesh(msh, R.as_matrix())

    overhang, plane, SV = construct_support_volume(msh, thresh, plane_offset)

    # now subtract the volume caused by the offset
    pts = overhang.project_points_to_plane(origin=plane.center)
    V_offset = pts.area * plane_offset

    return -(SV.volume - V_offset)


def main_pyvista():
    # set parameters
    OVERHANG_THRESHOLD = 0.0
    PLANE_OFFSET = 0.0
    NUM_START = 1
    GRID = True
    FILE = 'Geometries/cube.stl'

    # create mesh and clean
    mesh = pv.read(FILE)
    mesh = prep_mesh(mesh)

    # perform grid search
    if GRID:
        print(f'Perform grid search and extract {NUM_START} max values')
        ax, ay, f = grid_search_pyvista(mesh, max_angle=np.deg2rad(180), plot=True)
        flat_idx = np.argpartition(f.ravel(), -NUM_START)[-NUM_START:]
        row_idx, col_idx = np.unravel_index(flat_idx, f.shape)
        X0 = [[ax[row_idx[k]], ay[col_idx[k]]] for k in range(NUM_START)]
    else:
        X0 = [[np.deg2rad(30), np.deg2rad(40)]]

    # run optimizer for every start
    res = []
    for i in range(NUM_START):
        start = time()
        a0 = np.array(X0[i])
        print(f'Start #{i + 1} of optimizer, x0={np.rad2deg(a0)}')
        y = minimize(support_3D_pyvista, a0, jac='3-point',
                     args=(mesh, OVERHANG_THRESHOLD, PLANE_OFFSET))
        end = time()
        print(f'Computation time: {end - start} seconds')
        print(f'Optimization terminated with success: {y.success}')
        print(f'Maximum support volume of {-y.fun} at {np.rad2deg(y.x)} degrees')
        res.append(y)

    # create a pv Plotter and show axis system
    plot = pv.Plotter()
    plot.add_axes()

    # reconstruct optimal orientation
    R = Rotation.from_euler('xyz', [y.x[0], y.x[1], 0])
    mesh_rot = rotate_mesh(mesh, R)
    overhang = extract_overhang(mesh_rot, OVERHANG_THRESHOLD)
    plane = construct_build_plane(mesh_rot, PLANE_OFFSET)
    SV = construct_supports(overhang, plane)

    # add original and rotated mesh, and support volume
    # plot.add_mesh(mesh, opacity=0.2, color='blue')
    plot.add_mesh(mesh_rot, color='green', opacity=0.5)
    plot.add_mesh(plane, color='purple', opacity=0.5)
    plot.add_mesh(SV, opacity=0.5, color='red', show_edges=True)
    plot.show()


def grid_search_pyvista(mesh=None, max_angle=np.deg2rad(90), num_it=21, plot=True,
                        overhang_threshold=0.0, plane_offset=0.0):
    # create mesh and clean
    if mesh is None:
        FILE = 'Geometries/cube.stl'
        mesh = pv.read(FILE)
        mesh = prep_mesh(mesh)

    # iteration parameters
    ax = ay = np.linspace(-max_angle, max_angle, num_it)
    f = np.zeros((ax.shape[0], ay.shape[0]))

    start = time()
    for i, x in enumerate(ax):
        for j, y in enumerate(ay):
            f[j, i] = -support_3D_pyvista([x, y], mesh, overhang_threshold, plane_offset)
    end = time()

    if plot:
        x, y = np.meshgrid(np.rad2deg(ax), np.rad2deg(ay))
        surf = pv.StructuredGrid(x, y, f)
        surf_plot = pv.Plotter()
        surf_plot.add_mesh(surf, scalars=surf.points[:, -1], show_edges=True,
                           scalar_bar_args={'vertical': True})
        surf_plot.set_scale(zscale=5)
        surf_plot.show_grid()

        opt_idx = np.unravel_index(np.argmax(f), f.shape)
        print(f'Execution time: {end - start} seconds')
        print(f'Max volume: {f[opt_idx]} at '
              f'{round(x[opt_idx], 1), round(y[opt_idx], 1)} degrees')
        surf_plot.show()

    return ax, ay, f


def calc_V(points, d):
    p1, p2, p3 = points[0, :], points[1, :], points[2, :]
    v1 = p1[0] * p2[1] - p2[0] * p1[1]
    v2 = p2[0] * p3[1] - p3[0] * p2[1]
    v3 = p3[0] * p1[1] - p1[0] * p3[1]
    A = abs(v1 + v2 + v3) / 2
    h = (p1[-1] + p2[-1] + p3[-1]) / 3 - d
    return A, h


def calc_dVdx(points: np.ndarray, dpoints: np.ndarray, area: float, height: float) -> float:

    def det(x1, y1, x2, y2):
        return x1*y2 - y1*x2

    # extract points and derivatives
    p1, p2, p3 = points[0, :], points[1, :], points[2, :]
    dp1, dp2, dp3 = dpoints[0, :], dpoints[1, :], dpoints[2, :]
    #
    # det1 = det(dp1[0], p1[1], dp2[0], p2[1])
    # det2 = det(p1[0], dp1[1], p2[0], dp2[1])
    # det3 = det(dp2[0], p2[1], dp3[0], p3[1])
    # det4 = det(p2[0], dp2[1], p3[0], dp3[1])
    # det5 = det(dp3[0], p3[1], dp1[0], p1[1])
    # det6 = det(p3[0], dp3[1], p1[0], dp1[1])
    #
    # dAdx = sum([det1, det2, det3, det4, det5, det6])/2
    dx1 = (p2[1] - p3[1]) * dp1[0]
    dx2 = (-p1[1] + p3[1]) * dp2[0]
    dx3 = (-p2[1] + p1[1]) * dp3[0]
    dy1 = (-p2[0] + p3[0]) * dp1[1]
    dy2 = (p1[0] - p3[0]) * dp2[1]
    dy3 = (p2[0] - p1[0]) * dp3[1]
    dAdx = sum([dx1, dx2, dx3, dy1, dy2, dy3])/2
    dhdx = sum([dp1[-1], dp2[-1], dp3[-1]])/3
    return height*dAdx + area*dhdx


def construct_skew(x, y, z):
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def support_volume_analytic(angles: list, msh: pv.PolyData, thresh: float, plane_offset=1.0) -> tuple[float, list]:

    def cross_product(v1, v2) -> np.ndarray:
        return np.cross(v1, v2)

    # extract angles, construct rotation matrices for x and y rotations
    a, b = angles[0], angles[1]
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    R = Ry @ Rx

    # construct derivatives of rotation matrices
    dRx = construct_skew(1, 0, 0) @ Rx
    dRy = construct_skew(0, 1, 0) @ Ry
    dRda = Ry @ dRx
    dRdb = dRy @ Rx

    # rotate mesh
    msh_rot = rotate_mesh(msh, R)

    # define z-height of projection plane
    z_min = msh_rot.bounds[-2] - plane_offset

    # extract overhanging faces
    overhang_idx = np.arange(msh.n_cells)[msh_rot['Normals'][:, 2] < thresh]

    volume = 0.0
    dVda = 0.0
    dVdb = 0.0
    i = 0
    for idx in overhang_idx:
        pts = msh_rot.extract_cells(idx).points
        pts0 = msh.extract_cells(idx).points

        # check if cell connectivity is counterclockwise, if not switch p2 and p3
        n_check = cross_product((pts[1, :] - pts[0, :]), (pts[2, :] - pts[0, :]))
        if not np.allclose(msh_rot.cell_normals[idx], n_check/np.linalg.norm(n_check)):
            pts = np.array([pts[0, :], pts[2, :], pts[1, :]])
            pts0 = np.array([pts0[0, :], pts0[2, :], pts0[1, :]])

        # calculate support volume
        A, h = calc_V(pts, z_min)
        volume += A*h

        # rotate points to obtain derivatives
        # # transpose to reflect notation of pts -> points are sorted row wise
        dpts_da = np.transpose(dRda @ np.transpose(pts0))
        dpts_db = np.transpose(dRdb @ np.transpose(pts0))

        # calculate derivatives
        dVda_ = calc_dVdx(pts, dpts_da, A, h)
        # print(f'i: {i}')
        # dVdb_ = calc_dVdx(pts, dpts_db, A, h)

        dVda += dVda_
        # dVdb += dVdb_
        i += 1

    return -volume, [dVda, dVdb]


def main_analytic():
    # set parameters
    OVERHANG_THRESHOLD = -1e-5
    PLANE_OFFSET = 0
    NUM_START = 4
    GRID = True
    FILE = 'Geometries/cube.stl'

    # create mesh and clean
    mesh = pv.read(FILE)
    mesh = prep_mesh(mesh)

    angles = np.linspace(np.deg2rad(10), np.deg2rad(80), 10)
    f = []
    da = []
    db = []
    for a in angles:
        f_, [da_, db_] = support_volume_analytic([a, 0], mesh, OVERHANG_THRESHOLD, PLANE_OFFSET)
        f.append(f_)
        da.append(da_)
        db.append(db_)

    _ = plt.plot(angles, f, 'b', label='Volume')
    _ = plt.plot(angles, da, 'r', label='dVda')
    _ = plt.plot(angles, db, 'g', label='dVdb')
    _ = plt.legend()
    plt.show()

    a = np.array(np.deg2rad([10, 10]))
    start = time()
    y = minimize(support_volume_analytic, a, jac=True,
                 args=(mesh, OVERHANG_THRESHOLD, PLANE_OFFSET))
    end = time() - start
    print(y)
    print(f'Optimal orientation at {np.rad2deg(y.x)}')
    print(f'Computation time: {end} s')


if __name__ == "__main__":
    main_analytic()
