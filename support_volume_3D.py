import numpy as np
import pyvista as pv
from time import time
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from helpers import *


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
        make_surface_plot(np.rad2deg(ax), np.rad2deg(ay), f)
        opt_idx = np.unravel_index(np.argmax(f), f.shape)
        print(f'Execution time: {end - start} seconds')
        print(f'Max volume: {f[opt_idx]} at '
              f'{round(x[opt_idx], 1), round(y[opt_idx], 1)} degrees')

    return ax, ay, f


def support_volume_analytic(angles: list, msh: pv.PolyData, func_args) -> tuple[float, list]:
    thresh, plane = func_args

    # extract angles, construct rotation matrices for x and y rotations
    Rx, Ry, R, dRda, dRdb = construct_rotation_matrix(angles[0], angles[1])

    # rotate mesh
    msh_rot = rotate_mesh(msh, R)

    # define z-height of projection plane for adaptive projection
    # z_min = msh_rot.points[np.argmin(msh_rot.points[:, -1]), :]
    # dzda = dRda @ rotate2initial(z_min, R)
    # dzdb = dRdb @ rotate2initial(z_min, R)

    # define z-height of projection plane for fixed projection height
    z_min = np.array([0, 0, -plane])
    dzda = dzdb = [0]

    # extract overhanging faces
    overhang_idx = np.arange(msh.n_cells)[msh_rot['Normals'][:, 2] < thresh]

    build_dir = np.array([0, 0, 1])
    volume = 0.0
    dVda = 0.0
    dVdb = 0.0
    for idx in overhang_idx:

        # extract points and normal vector from cell
        cell = msh_rot.extract_cells(idx)
        points = np.transpose(cell.points)
        normal = np.transpose(msh_rot['Normals'][idx])

        # normal has to be of unit length for area calculation
        normal /= np.linalg.norm(normal)

        # compute initial points and normal vector
        points0 = rotate2initial(points, R)
        normal0 = msh['Normals'][idx]

        # calculate area and height
        A = cell.area * -build_dir.dot(normal)
        h = sum(points[-1]) / 3 - z_min[-1]
        volume += A * h

        # calculate area derivative
        dAda = cell.area * -build_dir.dot(dRda @ normal0)
        dAdb = cell.area * -build_dir.dot(dRdb @ normal0)

        # calculate height derivative
        dhda = sum((dRda @ points0)[-1]) / 3 - dzda[-1]
        dhdb = sum((dRdb @ points0)[-1]) / 3 - dzdb[-1]

        # calculate volume derivative and sum
        dVda_ = A * dhda + h * dAda
        dVdb_ = A * dhdb + h * dAdb
        dVda += dVda_
        dVdb += dVdb_

    return -volume, [-dVda, -dVdb]


def support_volume_smooth(angles: list, msh: pv.PolyData, func_args) -> tuple[float, list]:
    thresh, plane = func_args

    # extract angles, construct rotation matrices for x and y rotations
    Rx, Ry, R, dRda, dRdb = construct_rotation_matrix(angles[0], angles[1])

    # rotate mesh
    msh_rot = rotate_mesh(msh, R)

    # define z-height of projection plane for fixed projection height
    z_min = np.array([0, 0, -plane])
    dzda = dzdb = [0]

    build_dir = np.array([0, 0, 1])
    volume = 0.0
    dVda = 0.0
    dVdb = 0.0
    for idx in range(msh_rot.n_cells):

        # extract points and normal vector from cell
        cell = msh_rot.extract_cells(idx)
        points = np.transpose(cell.points)
        normal = np.transpose(cell.cell_data.active_normals[0])

        # normal has to be of unit length for area calculation
        normal /= np.linalg.norm(normal)

        # compute initial points and normal vector
        points0 = rotate2initial(points, R)
        normal0 = rotate2initial(normal, R)

        # calculate derivative of normal
        dnda = dRda @ normal0
        dndb = dRdb @ normal0

        # calculate the smooth Heaviside of the normal and its derivative
        k = 10
        H = smooth_heaviside(-1 * normal[-1], k, thresh)
        dHda = H * (1 - H) * 2 * k * -dnda[-1]
        dHdb = H * (1 - H) * 2 * k * -dndb[-1]

        # calculate area and height
        A = cell.area * -build_dir.dot(normal)
        h = sum(points[-1]) / 3 - z_min[-1]
        volume += H * A * h

        # calculate area derivative
        dAda = cell.area * -build_dir.dot(dnda)
        dAdb = cell.area * -build_dir.dot(dndb)

        # calculate height derivative
        dhda = sum((dRda @ points0)[-1]) / 3 - dzda[-1]
        dhdb = sum((dRdb @ points0)[-1]) / 3 - dzdb[-1]

        # calculate volume derivative and sum
        dVda_ = H * A * dhda + H * h * dAda + dHda * A * h
        dVdb_ = H * A * dhdb + H * h * dAdb + dHdb * A * h
        dVda += dVda_
        dVdb += dVdb_

    return -volume, [-dVda, -dVdb]


def main_analytic():
    # set parameters
    OVERHANG_THRESHOLD = 0
    NUM_START = 1
    GRID = False
    MAX_ANGLE = 180
    # FILE = 'Geometries/cube.stl'

    # create mesh and clean
    # mesh = pv.read(FILE)
    # mesh = prep_mesh(mesh)
    # points = np.array([[-1 / 2, -np.sqrt(3) / 6, 0], [1 / 2, -np.sqrt(3) / 6, 0], [0, np.sqrt(3) / 3, 0]])
    # mesh = prep_mesh(pv.Triangle(points), flip=True)  # flip normal to ensure downward facing
    cube = pv.Cube()
    mesh = prep_mesh(cube)

    # set fixed projection distance
    PLANE_OFFSET = calc_min_projection_distance(mesh)
    #
    # angles = np.linspace(np.deg2rad(-MAX_ANGLE), np.deg2rad(-MAX_ANGLE), 101)
    # f = []
    # da = []
    # db = []
    # for a in angles:
    #     f_, [da_, db_] = support_volume_smooth([a, 0], mesh, OVERHANG_THRESHOLD, PLANE_OFFSET)
    #     f.append(-f_)
    #     da.append(-da_)
    #     db.append(-db_)

    angles, f, da, db = grid_search_1D(support_volume_smooth, mesh, OVERHANG_THRESHOLD, PLANE_OFFSET, MAX_ANGLE, 201)
    f = -f
    da = -da
    db = -db

    _ = plt.plot(np.rad2deg(angles), f, 'g', label='Volume')
    _ = plt.plot(np.rad2deg(angles), da, 'b', label=r'$V_{,\alpha}$')
    _ = plt.plot(np.rad2deg(angles), db, 'k', label=r'$V_{,\beta}$')
    _ = plt.plot(np.rad2deg(angles)[:-1], finite_forward_differences(f, angles), 'r.', label='Finite differences')
    plt.xlabel('Angle [deg]')
    # plt.ylim([-0.3, 0.3])
    plt.title(f'3D cube with 60 deg overhang threshold - rotation about x-axis')
    _ = plt.legend()
    # plt.savefig('out/supportvolume/3D_cube_rotx_60deg_smooth.svg', format='svg', bbox_inches='tight')
    plt.show()

    # perform grid search
    # if GRID:
    #     ax, ay, f = grid_search(support_volume_analytic, mesh, OVERHANG_THRESHOLD, PLANE_OFFSET)
    #     x0 = extract_x0(ax, ay, f, NUM_START)
    #
    #     make_contour_plot(np.rad2deg(ax), np.rad2deg(ay), -f, 'Unit cube contour plot', 'out/supportvolume/3D_cube_contour.svg')
    # else:
    #     x0 = [np.deg2rad([5, 5])]
    #
    # res = []
    # for i in range(NUM_START):
    #     start = time()
    #
    #     # set initial condition
    #     a = np.array(x0[i])
    #     print(f'Iteration {i + 1} with x0: {np.rad2deg(a)} degrees')
    #
    #     y = minimize(support_volume_analytic, a, jac=True,
    #                  args=(mesh, OVERHANG_THRESHOLD, PLANE_OFFSET))
    #     end = time() - start
    #     print(y)
    #     print(f'Optimal orientation at {np.rad2deg(y.x)} degrees')
    #     print(f'Computation time: {end} s')
    #     res.append(y)

    print('Finished')


if __name__ == "__main__":
    main_analytic()