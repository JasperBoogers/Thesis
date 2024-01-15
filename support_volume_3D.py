import numpy as np
import pyvista as pv
import pymeshfix as mf
from time import time
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from pyvista_functions import *


def support_3D_pyvista(angles: list, msh: pv.PolyData, thresh: float, plane_offset=1.0) -> float:
    # rotate
    R = Rotation.from_euler('xyz', np.append(angles, 0))
    msh = rotate_mesh(msh, R)

    overhang, plane, SV = construct_support_volume(msh, thresh, plane_offset)

    # now subtract the volume caused by the offset
    pts = overhang.project_points_to_plane(origin=plane.center)
    V_offset = pts.area * plane_offset

    return -(SV.volume - V_offset)


def main_pyvista():
    # set parameters
    OVERHANG_THRESHOLD = 0.0
    PLANE_OFFSET = 1.0
    NUM_START = 4
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
        X0 = [np.deg2rad(1), np.deg2rad(1)]

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
                        overhang_threshold=0.0, plane_offset=1.0):
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


def support_volume_analytic(angles: list, msh: pv.PolyData, thresh: float, plane_offset=1.0) -> float:
    # rotate the mesh
    R = Rotation.from_euler('xyz', np.append(angles, 0))
    msh = rotate_mesh(msh, R)

    # define z-height of projection plane
    z_min = msh.bounds[-2] - plane_offset

    # extract overhanging faces
    overhang_idx = np.arange(msh.n_cells)[msh['Normals'][:, 2] < thresh]

    volume = 0
    for idx in overhang_idx:
        # overhang = msh.extract_cells(idx).extract_surface()
        #
        # # compute area of projected triangle
        # area = overhang.project_points_to_plane(origin=[0, 0, z_min]).triangulate().area
        #
        # # extract cell center coordinates
        # centroid = overhang.cell_centers().points[0]
        #
        # # compute volume and add to total
        # volume += area * (centroid[-1] - z_min)

        pts = msh.extract_cells(idx).points

        v1 = pts[0, 0] * pts[1, 1] - pts[1, 0] * pts[0, 1]
        v2 = pts[1, 0] * pts[2, 1] - pts[2, 0] * pts[1, 1]
        v3 = pts[2, 0] * pts[0, 1] - pts[0, 0] * pts[2, 1]
        volume += abs(v1 + v2 + v3)*sum(pts[:, 2] - z_min)/6

    return -volume


def main_analytic():
    # set parameters
    OVERHANG_THRESHOLD = 0.0
    PLANE_OFFSET = 0
    NUM_START = 4
    GRID = True
    FILE = 'Geometries/cube.stl'

    # create mesh and clean
    mesh = pv.read(FILE)
    mesh = prep_mesh(mesh)

    a = np.array([30, 40])
    start = time()
    y = minimize(support_volume_analytic, a, jac='3-point',
                 args=(mesh, OVERHANG_THRESHOLD, PLANE_OFFSET))
    end = time() - start


if __name__ == "__main__":
    main_analytic()
