import pyvista as pv
import numpy as np
from helpers import *


def SoP_top_cover(angles: list, msh: pv.PolyData, thresh: float, plane: float) -> tuple[float, list]:

    # extract angles, construct rotation matrices for x and y rotations
    Rx, Ry, R, dRdx, dRdy = construct_rotation_matrix(angles[0], angles[1])

    # rotate mesh
    msh_rot = rotate_mesh(msh, R)

    # define z-height of projection plane for fixed projection height
    z_min = np.array([0, 0, -plane])

    # extract overhanging faces
    upward_idx = np.arange(msh.n_cells)[msh_rot['Normals'][:, 2] > thresh]
    top_cover, lines = extract_top_cover(msh_rot.extract_cells(upward_idx))

    build_dir = np.array([0, 0, 1])
    volume = 0.0
    dVda = 0.0
    dVdb = 0.0
    for idx in range(top_cover.n_cells):

        # extract points and normal vector from cell
        cell = top_cover.extract_cells(idx)
        vol, dVda_, dVdb_ = calc_V_under_triangle(cell, angles, build_dir, z_min)
        volume += vol
        dVda += dVda_
        dVdb += dVdb_

    return -(volume - msh.volume), [-dVda, -dVdb]


def plot_grid_sampling(mesh):
    # extract upward facing triangles
    idx = np.arange(mesh.n_cells)[mesh['Normals'][:, 2] > 1e-6]
    upward = mesh.extract_cells(idx)
    top_cover, lines = extract_top_cover(upward)

    # plot
    p = pv.Plotter()
    _ = p.add_mesh(upward, show_edges=True, color='g', opacity=0.5)
    _ = p.add_mesh(top_cover, show_edges=True, color='r', opacity=0.5)
    for line in lines:
        _ = p.add_mesh(line, color='b')
    p.show()

if __name__ == '__main__':

    # load file and rotate
    OVERHANG_THRESHOLD = 1e-5
    FILE = 'Geometries/chair.stl'
    m = pv.read(FILE)
    m = prep_mesh(m)

    # showcase of grid sampling
    # plot_grid_sampling(m.rotate_x(45, inplace=False))

    # set fixed projection distance
    PLANE_OFFSET = calc_min_projection_distance(m)

    ang = np.linspace(np.deg2rad(-180), np.deg2rad(180), 101)
    f = []
    da = []
    db = []
    for a in ang:
        f_, [da_, db_] = SoP_top_cover([a, 0], m, OVERHANG_THRESHOLD, PLANE_OFFSET)
        f.append(-f_)
        da.append(-da_)
        db.append(-db_)

    _ = plt.plot(np.rad2deg(ang), f, 'g', label='Volume')
    _ = plt.plot(np.rad2deg(ang), da, 'b', label=r'dV/d$\theta_x$')
    _ = plt.plot(np.rad2deg(ang), db, 'k', label=r'dV/d$\theta_y$')
    _ = plt.plot(np.rad2deg(ang)[:-1], finite_forward_differences(f, ang), 'r', label='Finite differences')
    plt.xlabel('Angle [deg]')
    # plt.ylim([-0.3, 0.3])
    plt.title(f'Chair with fixed projection to y=-{PLANE_OFFSET} - rotation about x-axis')
    _ = plt.legend()
    # plt.savefig('out/supportvolume/3D_cube_rotx_fixed_proj.svg', format='svg', bbox_inches='tight')
    plt.show()

    print('')
