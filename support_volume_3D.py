import numpy as np
import pyvista as pv
from time import time
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation


def prep_mesh(m, decimation=0.9, flip=False) -> pv.PolyData:
    # ensure mesh is only triangles
    m.triangulate(inplace=True)

    # decimate mesh by decimate*100%
    m.decimate_pro(decimation)

    # (re)compute normals, and flip normal direction if needed
    m.compute_normals(inplace=True, flip_normals=flip)

    # move mesh center of bounding box to origin
    center = np.array(m.center)
    m.translate(-center, inplace=True)
    return m


def rotate_mesh(m, rot: Rotation) -> pv.PolyData:
    # Rotate the mesh through the rotation obj R
    tfm = np.identity(4)
    tfm[:-1, :-1] = rot.as_matrix()
    return m.transform(tfm, inplace=False)


def extract_overhang(m, t) -> pv.PolyData:
    idx = np.arange(m.n_cells)[m['Normals'][:, 2] < t]
    overhang = m.extract_cells(idx)
    return overhang.extract_surface()


def construct_build_plane(m, offset) -> pv.PolyData:
    bounds = m.bounds
    return pv.Plane(center=(0, 0, bounds[-2] - offset),
                    i_size=1.1*(bounds[1] - bounds[0]),
                    j_size=1.1*(bounds[3] - bounds[2]),
                    direction=(0, 0, 1))


def construct_supports(o, p) -> pv.PolyData:
    SV = o.extrude_trim((0, 0, -1), p)
    SV.triangulate(inplace=True)
    SV.compute_normals(inplace=True, flip_normals=True)
    return SV


def support_3D_Euler(angles, msh, thresh, plane_offset=1.0) -> float:

    # rotate
    R = Rotation.from_euler('xyz', np.append(angles, 0))
    msh = rotate_mesh(msh, R)

    # extract overhanging surfaces
    overhang = extract_overhang(msh, thresh)

    # construct print bed plane based on lowest mesh point,
    # add an offset to ensure proper triangulation
    plane = construct_build_plane(msh, plane_offset)

    # extrude overhanging surfaces to projection plane
    SV = construct_supports(overhang, plane)

    # now subtract the volume caused by the offset
    pts = overhang.project_points_to_plane(origin=plane.center)
    V_offset = pts.area*plane_offset

    return -(SV.volume-V_offset)


def main():
    # set parameters
    OVERHANG_THRESHOLD = 0.0
    PLANE_OFFSET = 1.0
    FILE = 'Geometries/cube.stl'

    # create mesh and clean
    mesh = pv.read(FILE)
    mesh = prep_mesh(mesh)

    # optimize
    a0 = np.array([np.deg2rad(1), np.deg2rad(1)])
    start = time()

    y = minimize(support_3D_Euler, a0, jac='3-point',
                 args=(mesh, OVERHANG_THRESHOLD, PLANE_OFFSET))
    end = time()
    print(f'Computation time: {end-start} seconds')
    print(f'Optimization terminated with success: {y.success}')
    print(f'Maximum support volume of {-y.fun} at {np.rad2deg(y.x)} degrees')

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


def grid_search():
    # set parameters
    OVERHANG_THRESHOLD = 0.0
    PLANE_OFFSET = 1.0
    FILE = 'Geometries/cube.stl'

    # create mesh and clean
    mesh = pv.read(FILE)
    mesh = prep_mesh(mesh)

    # iteration parameters
    MAX_ANGLE = np.deg2rad(90)
    NUM_IT = 21
    ax = ay = np.linspace(-MAX_ANGLE, MAX_ANGLE, NUM_IT)
    f = np.zeros((ax.shape[0], ay.shape[0]))

    start = time()
    for i, x in enumerate(ax):
        for j, y in enumerate(ay):
            rot = rotate_mesh(mesh, [x, y, 0])
            overhang = extract_overhang(rot, OVERHANG_THRESHOLD)
            plane = construct_build_plane(rot, PLANE_OFFSET)
            SV = construct_supports(overhang, plane)

            pts = overhang.project_points_to_plane(origin=plane.center)
            V_offset = pts.area*PLANE_OFFSET
            f[j, i] = SV.volume-V_offset
    end = time()

    x, y = np.meshgrid(np.rad2deg(ax), np.rad2deg(ay))
    surf = pv.StructuredGrid(x, y, f)
    surf_plot = pv.Plotter()
    surf_plot.add_mesh(surf, scalars=surf.points[:, -1], show_edges=True,
                       scalar_bar_args={'vertical': True})
    surf_plot.set_scale(zscale=5)
    surf_plot.show_grid()

    opt_idx = np.unravel_index(np.argmax(f), f.shape)
    print(f'Execution time: {end-start} seconds')
    print(f'Max volume: {f[opt_idx]} at '
          f'{round(x[opt_idx], 1), round(y[opt_idx], 1)} degrees')
    surf_plot.show()


if __name__ == "__main__":
    # grid_search()
    main()
