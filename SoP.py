import time
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

    # extract upward facing facets
    upward_idx = np.arange(msh_rot.n_cells)[msh_rot['Normals'][:, 2] > thresh]
    top_cover, _ = extract_top_cover(msh_rot.extract_cells(upward_idx))

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

    return -(volume - msh_rot.volume), [-dVda, -dVdb]


def plot_top_cover(mesh, angle=0):
    # extract upward facing triangles
    idx = np.arange(mesh.n_cells)[mesh['Normals'][:, 2] > 1e-6]
    upward = mesh.extract_cells(idx)
    top_cover, lines = extract_top_cover(upward)

    plot_intermediate(mesh, upward, top_cover, lines, angle)


def plot_correction_facets(mesh, angle=0):

    # compute average coordinate for each cell, and store in 'Center' array
    mesh.cell_data['Center'] = [np.sum(c.points, axis=0) / 3 for c in mesh.cell]

    # extract overhang
    overhang_idx = np.arange(mesh.n_cells)[mesh['Normals'][:, 2] < -1e-6]
    overhang = mesh.extract_cells(overhang_idx)

    # extract correction
    correction_idx, lines = extract_correction_idx(mesh, overhang_idx)

    if len(correction_idx) > 0:
        correction_facets = mesh.extract_cells(correction_idx)
    else:
        correction_facets = None
    plot_intermediate(mesh, overhang, correction_facets, lines, angle)


def plot_intermediate(mesh, selection, correction, lines, angle):

    p = pv.Plotter()
    _ = p.add_mesh(mesh, style='wireframe', color='k', show_edges=True)
    _ = p.add_mesh(selection, show_edges=True, color='b', opacity=1)
    if correction is not None:
        _ = p.add_mesh(correction, show_edges=True, color='r', opacity=1)
    for line in lines:
        _ = p.add_mesh(line, color='y')
    _ = p.add_text(f'rotation of {angle} degrees')
    p.show()


def SoP_naive_correction(angles: list, msh: pv.PolyData, thresh: float, plane: float) -> tuple[float, list]:

    # extract angles, construct rotation matrices for x and y rotations
    Rx, Ry, R, dRdx, dRdy = construct_rotation_matrix(angles[0], angles[1])

    # rotate mesh
    msh_rot = rotate_mesh(msh, R)

    # compute average coordinate for each cell, and store in 'Center' array
    msh_rot.cell_data['Center'] = [np.sum(c.points, axis=0) / 3 for c in m.cell]

    # extract overhanging faces
    overhang_idx = np.arange(msh_rot.n_cells)[msh_rot['Normals'][:, 2] < -thresh]
    correction_idx, lines = extract_correction_idx(msh_rot, overhang_idx)

    # define z-height of projection plane for fixed projection height
    z_min = np.array([0, 0, -plane])

    build_dir = np.array([0, 0, 1])
    volume = 0.0
    dVda = 0.0
    dVdb = 0.0

    for idx in overhang_idx:
        # extract points and normal vector from cell
        cell = msh_rot.extract_cells(idx)
        vol, dVda_, dVdb_ = calc_V_under_triangle(cell, angles, -build_dir, z_min)
        volume += vol
        dVda += dVda_
        dVdb += dVdb_

    for idx in correction_idx:
        cell = msh_rot.extract_cells(idx)
        vol, dVda_, dVdb_ = calc_V_under_triangle(cell, angles, build_dir, z_min)
        volume -= vol
        dVda -= dVda_
        dVdb -= dVdb_

    return -volume, [-dVda, -dVdb]


if __name__ == '__main__':

    # load file and rotate
    OVERHANG_THRESHOLD = 1e-5
    FILE = 'Geometries/cube_cutout.stl'

    m = pv.read(FILE)
    # m = pv.Cube()
    m = prep_mesh(m, decimation=0)
    m = m.subdivide(2, subfilter='linear')

    # set fixed projection distance
    PLANE_OFFSET = calc_min_projection_distance(m)
    # ang = np.linspace(np.deg2rad(-180), np.deg2rad(180), 201)
    # f = []
    # da = []
    # db = []

    start = time.time()
    # for a in ang:
    #     f_, [da_, db_] = SoP_naive_correction([a, 0], m, OVERHANG_THRESHOLD, PLANE_OFFSET)
    #     f.append(-f_)
    #     da.append(-da_)
    #     db.append(-db_)

    ang, f, da, db = grid_search_1D(SoP_top_cover, m, OVERHANG_THRESHOLD, PLANE_OFFSET, np.pi, 101, 'x')
    f = -f
    da = -da
    db = -db
    # ax, ay, f = grid_search(SoP_top_cover, m, OVERHANG_THRESHOLD, PLANE_OFFSET, np.deg2rad(180), 20)

    _ = plt.plot(np.rad2deg(ang), -f, 'g', label='Volume')
    _ = plt.plot(np.rad2deg(ang), da, 'b.', label=r'$V_{,\alpha}$')
    _ = plt.plot(np.rad2deg(ang), db, 'k.', label=r'$V_{,\beta}$')
    _ = plt.plot(np.rad2deg(ang)[:-1], finite_forward_differences(f, ang), 'r.', label='Finite differences')
    plt.xlabel('Angle [deg]')
    plt.ylim([-2, 2])
    plt.title(f'Cube with cutout - rotation about x-axis, Ezair method')
    _ = plt.legend()
    # plt.savefig('out/supportvolume/SoP_cube_rotx_correction.svg', format='svg', bbox_inches='tight')
    plt.show()

    end = time.time()
    print(f'Finished in {end-start} seconds')
    # make_contour_plot(np.rad2deg(ax), np.rad2deg(ay), f, 'Reference - unit cube')

    print()
