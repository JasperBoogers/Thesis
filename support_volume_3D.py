import numpy as np
import pyvista as pv
from time import time
from os import path
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


def rotate_mesh(m, a):
    # create rotation obj and rotate the mesh
    tfm = np.identity(4)
    rot = Rotation.from_euler('xyz', a)
    tfm[:-1, :-1] = rot.as_matrix()
    return m.transform(tfm, inplace=False)


def extract_overhang(m, t):
    idx = np.arange(m.n_cells)[m['Normals'][:, 2] < t]
    overhang = m.extract_cells(idx)
    return overhang.extract_surface()


def construct_build_plane(m, offset):
    bounds = m.bounds
    return pv.Plane(center=(0, 0, bounds[-2] - offset),
                    i_size=1.1*(bounds[1] - bounds[0]),
                    j_size=1.1*(bounds[3] - bounds[2]),
                    direction=(0, 0, 1))


def construct_supports(o, p):
    SV = o.extrude_trim((0, 0, -1), p)
    SV.triangulate(inplace=True)
    SV.compute_normals(inplace=True, flip_normals=True)
    return SV


def support_3D_Euler(angles, msh, thresh, plane_offset=1.0) -> float:

    # rotate
    msh = rotate_mesh(msh, np.append(angles, [0, 0]))

    # extract overhanging surfaces
    overhang = extract_overhang(msh, thresh)

    # construct print bed plane based on lowest mesh point,
    # add a offset to ensure proper triangulation
    plane = construct_build_plane(msh, plane_offset)

    # extrude overhanging surfaces to projection plane
    SV = construct_supports(overhang, plane)

    # now subtract the volume caused by the offset
    b = SV.bounds
    V_offset = (b[1] - b[0]) * (b[3] -b[2]) * (msh.bounds[-2] - b[-2])

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
    a0 = [np.pi/5]
    start = time()

    y = minimize(support_3D_Euler, a0,
                 args=(mesh, OVERHANG_THRESHOLD, PLANE_OFFSET))
    end = time()
    print(f'Computation time: {end-start} seconds')
    print(f'Optimization terminated with succes: {y.success}')
    print(f'Maximum support volume of {-y.fun} at {np.rad2deg(y.x)} degrees')

    # create a pv Plotter and show axis system
    plot = pv.Plotter()
    plot.add_axes()

    # reconstruct optimal orientation
    mesh_rot = rotate_mesh(mesh, np.append(y.x, [0, 0]))
    overhang = extract_overhang(mesh_rot, OVERHANG_THRESHOLD)
    plane = construct_build_plane(mesh_rot, PLANE_OFFSET)
    SV = construct_supports(overhang, plane)

    # add original and rotated mesh, and support volume
    plot.add_mesh(mesh, opacity=0.2, color='green')
    plot.add_mesh(mesh_rot, color='blue')
    plot.add_mesh(SV, opacity=0.5, color='red', show_edges=True)
    plot.show(interactive_update=True)
    print('finish')


if __name__ == "__main__":
    main()
