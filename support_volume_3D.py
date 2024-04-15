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


def support_volume_analytic(angles: list, msh: pv.PolyData, par) -> tuple[float, list]:
    thresh = par['down_thresh']
    plane = par['plane_offset']

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

    # extract normal vectors
    M = msh_rot['Normals'][:, 2] < thresh

    # calculate projected area and height of each facet
    A, dAda, dAdb, h, dhda, dhdb = calc_V_vectorized(msh, msh_rot, dRda, dRdb, z_min, dzda, dzdb, par)

    volume = sum(M * A * h)
    dVda = np.sum(M * A * dhda + M * dAda * h)
    dVdb = np.sum(M * A * dhdb + M * dAdb * h)

    return -volume, [-dVda, -dVdb]


def support_volume_smooth(angles: list, msh: pv.PolyData, par: dict) -> tuple[float, list]:
    thresh = par['down_thresh']
    plane = par['plane_offset']
    k_down = par['down_k']

    # extract angles, construct rotation matrices for x and y rotations
    Rx, Ry, R, dRda, dRdb = construct_rotation_matrix(angles[0], angles[1])

    # rotate mesh
    msh_rot = rotate_mesh(msh, R)

    z_min, dz_min = mellow_min(msh_rot.points, -80)
    dzda = np.sum(dz_min * np.transpose(dRda @ np.transpose(msh.points)), axis=0)
    dzdb = np.sum(dz_min * np.transpose(dRdb @ np.transpose(msh.points)), axis=0)

    # extract normal vectors
    normals = msh_rot['Normals']
    normals0 = msh['Normals']

    # derivative of normals
    dnda = np.transpose(dRda @ np.transpose(normals0))
    dndb = np.transpose(dRdb @ np.transpose(normals0))

    # calculate smooth heaviside of the normals
    M = smooth_heaviside(-1 * normals[:, 2], k_down, thresh)
    dMda = M * (1 - M) * 2 * k_down * -dnda[:, 2]
    dMdb = M * (1 - M) * 2 * k_down * -dndb[:, 2]

    # calculate projected area and height of each facet
    A, dAda, dAdb, h, dhda, dhdb = calc_V_vectorized(msh, msh_rot, dRda, dRdb, z_min, dzda, dzdb, par)

    volume = sum(M * A * h)
    dVda = np.sum(M * A * dhda + M * dAda * h + dMda * A * h)
    dVdb = np.sum(M * A * dhdb + M * dAdb * h + dMdb * A * h)

    return volume, [dVda, dVdb]


def main_analytic():
    # set parameters
    NUM_START = 1
    GRID = False
    MAX_ANGLE = np.deg2rad(180)
    # FILE = 'Geometries/cube.stl'

    # create mesh and clean
    # mesh = pv.read(FILE)
    # mesh = prep_mesh(mesh)
    # points = np.array([[-1 / 2, -np.sqrt(3) / 6, 0], [1 / 2, -np.sqrt(3) / 6, 0], [0, np.sqrt(3) / 3, 0]])
    # mesh = prep_mesh(pv.Triangle(points), flip=True)  # flip normal to ensure downward facing
    mesh = pv.Cube()
    mesh = prep_mesh(mesh, decimation=0)
    mesh = mesh.subdivide(2, 'linear')
    mesh = prep_mesh(mesh, decimation=0)

    args = {
        'build_dir': np.array([0, 0, 1]),
        'down_thresh': np.sin(np.deg2rad(0)),
        'up_thresh': np.sin(np.deg2rad(0)),
        'down_k': 10,
        'up_k': 10,
        'plane_offset': calc_min_projection_distance(mesh),
        'SoP_penalty': 1
    }

    # angles = np.linspace(np.deg2rad(-MAX_ANGLE), np.deg2rad(-MAX_ANGLE), 101)
    # f = []
    # da = []
    # db = []
    # for a in angles:
    #     f_, [da_, db_] = support_volume_smooth([a, 0], mesh, args)
    #     f.append(-f_)
    #     da.append(-da_)
    #     db.append(-db_)

    angles, f, da, db = grid_search_1D(support_volume_smooth, mesh, args, MAX_ANGLE, 201)

    # args['plane_offset'] = 0
    #
    # a2, f2, da2, db2 = grid_search_1D(support_volume_smooth, mesh, args, MAX_ANGLE, 201)
    #
    # args['plane_offset'] = -1
    #
    # a3, f3, da3, db3 = grid_search_1D(support_volume_smooth, mesh, args, MAX_ANGLE, 201)

    _ = plt.plot(np.rad2deg(angles), f, 'g', label='Volume')
    _ = plt.plot(np.rad2deg(angles), da, 'b', label=r'dVda')
    _ = plt.plot(np.rad2deg(angles), db, 'k', label=r'dVdb')
    _ = plt.plot(np.rad2deg(angles), finite_central_differences(f, angles), 'r.', label='Finite differences')
    plt.xlabel('Angle [deg]')
    plt.ylabel(r'Volume [mm$^3$]')
    plt.title(f'3D cube - Comparison of projection method')
    _ = plt.legend()
    # plt.savefig('out/supportvolume/3D_cube_rotx_projection_comp.svg', format='svg')
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