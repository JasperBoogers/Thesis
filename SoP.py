import time
import pyvista as pv
import numpy as np
from helpers import *
from sensitivities import calc_cell_sensitivities, plot_cell_sensitivities


def SoP_top_cover(angles: list, msh: pv.PolyData, par: dict) -> tuple[float, list]:
    thresh = par['down_thresh']
    plane = par['plane_offset']

    # extract angles, construct rotation matrices for x and y rotations
    Rx, Ry, R, dRdx, dRdy = construct_rotation_matrix(angles[0], angles[1])

    # rotate mesh
    msh_rot = rotate_mesh(msh, R)

    # define z-height of projection plane for fixed projection height
    z_min = np.array([0, 0, -plane])

    # compute average coordinate for each cell, and store in 'Center' array
    msh_rot.cell_data['Center'] = [np.sum(c.points, axis=0) / 3 for c in msh_rot.cell]

    # extract upward facing facets
    upward_idx = np.arange(msh_rot.n_cells)[msh_rot['Normals'][:, 2] > thresh]
    top_idx, _ = extract_top_cover(msh_rot, upward_idx)

    build_dir = np.array([0, 0, 1])
    volume = 0.0
    dVda = 0.0
    dVdb = 0.0
    for idx in top_idx:  # TODO convert to calc_V_vectorized

        # extract points and normal vector from cell
        cell = msh_rot.extract_cells(idx)
        vol, dVda_, dVdb_ = calc_V_under_triangle(cell, angles, build_dir, z_min)
        volume += vol
        dVda += dVda_
        dVdb += dVdb_

    return -(volume - msh_rot.volume), [-dVda, -dVdb]


def SoP_top_smooth(angles: list, msh: pv.PolyData, par: dict) -> tuple[float, list]:
    thresh = par['down_thresh']
    plane = par['plane_offset']
    build_dir = par['build_dir']
    down_k = par['down_k']

    # extract angles, construct rotation matrices for x and y rotations
    Ra, Rb, R, dRda, dRdb = construct_rotation_matrix(angles[0], angles[1])

    # rotate mesh
    msh_rot = rotate_mesh(msh, R)

    # define z-height of projection plane for fixed projection height
    if plane == -1:
        z_min = msh_rot.points[np.argmin(msh_rot.points[:, -1]), :]
        dzda = dRda @ rotate2initial(z_min, R)
        dzdb = dRdb @ rotate2initial(z_min, R)
    else:
        z_min = np.array([0, 0, -plane])
        dzda = dzdb = [0]

    # compute average coordinate for each cell, and store in 'Center' array
    msh_rot.cell_data['Center'] = [np.sum(c.points, axis=0) / 3 for c in msh_rot.cell]

    # extract upward facing facets
    upward_idx = np.arange(msh_rot.n_cells)[msh_rot['Normals'][:, 2] > thresh]
    top_idx, _ = extract_top_cover(msh_rot, upward_idx)
    top = smooth_top_cover(msh_rot, par)

    volume = 0.0
    dVda = 0.0
    dVdb = 0.0

    for idx in range(msh_rot.n_cells):  # TODO convert to vectorized
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
        H = top[idx]
        dHda = H * (1 - H) * -2 * down_k * dnda[-1]
        dHdb = H * (1 - H) * -2 * down_k * dndb[-1]

        # calculate area and height
        A = cell.area * build_dir.dot(normal)
        h = sum(points[-1]) / 3 - z_min[-1]
        vol = H * A * h

        # calculate area derivative
        dAda = cell.area * build_dir.dot(dnda)
        dAdb = cell.area * build_dir.dot(dndb)

        # calculate height derivative
        dhda = sum((dRda @ points0)[-1]) / 3 - dzda[-1]
        dhdb = sum((dRdb @ points0)[-1]) / 3 - dzdb[-1]

        # calculate volume derivative and sum
        dVda_ = H * A * dhda + H * dAda * h + dHda * A * h
        dVdb_ = H * A * dhdb + H * dAdb * h + dHdb * A * h
        dVda += dVda_
        dVdb += dVdb_

        volume += vol
        dVda += dVda_
        dVdb += dVdb_

    return -(volume - msh_rot.volume), [-dVda, -dVdb]


def plot_top_cover(mesh, angle=0):
    # extract upward facing triangles
    idx = np.arange(mesh.n_cells)[mesh['Normals'][:, 2] > 1e-6]
    upward = mesh.extract_cells(idx)
    top_cover, lines = extract_top_cover(mesh, idx)

    plot_intermediate(mesh, upward, mesh.extract_cells(top_cover), lines, angle)


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
    _ = p.add_mesh(selection, show_edges=True, color='g', opacity=1)
    if correction is not None:
        _ = p.add_mesh(correction, show_edges=True, color='r', opacity=1)
    if lines is not None:
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


def smooth_overhang_mask_gif(mesh, filename):
    p = pv.Plotter(off_screen=True, notebook=False)
    p.add_axes()
    p.add_mesh(mesh, name='mesh', lighting=False,
               scalar_bar_args={"title": "Overhang value"}, clim=[-2, 1], cmap='brg', show_edges=True)
    p.show(interactive_update=True)

    p.open_gif(filename)
    n_frames = 120
    angles = np.deg2rad(np.linspace(180, 0, n_frames))
    for f in range(n_frames):
        # extract angles, construct rotation matrices for x and y rotations
        Ra, Rb, R, dRda, dRdb = construct_rotation_matrix(angles[f], 0)

        # rotate mesh
        mesh_rot = rotate_mesh(mesh, R)

        build_dir = np.array([0, 0, -1])

        # compute average coordinate for each cell, and store in 'Center' array
        mesh_rot.cell_data['Center'] = [np.sum(c.points, axis=0) / 3 for c in mesh_rot.cell]

        # compute overhang mask
        k = 10
        overhang = smooth_overhang(mesh_rot, build_dir, 1e-5, k)
        p.add_mesh(mesh_rot, scalars=overhang, name='mesh', lighting=False,
                   scalar_bar_args={"title": "Overhang value"}, clim=[-2, 1], cmap='brg', show_edges=True)
        p.update()
        p.write_frame()
        print(f'Writing frame for angle: {np.rad2deg(angles[f])} degrees')

    p.close()


def SoP_connectivity(angles: list, mesh: pv.PolyData, par) -> tuple[float, list]:
    plane = par['plane_offset']

    # extract angles, construct rotation matrices for x and y rotations
    Ra, Rb, R, dRda, dRdb = construct_rotation_matrix(angles[0], angles[1])

    # rotate mesh
    mesh_rot = rotate_mesh(mesh, R)

    # define z-height of projection plane for adaptive projection: lowest z-coordinate, closest to origin (xy norm)
    if plane <= 0:
        z_idx = np.where(mesh_rot.points[:, -1] == min(mesh_rot.points[:, -1]))[0]
        norm = np.linalg.norm(mesh_rot.points[z_idx, :-1], axis=1)
        min_norm_idx = np.argmin(norm)
        z_min = mesh_rot.points[z_idx[min_norm_idx], :] + np.array([0, 0, plane])
        dzda = dRda @ rotate2initial(z_min, R)
        dzdb = dRdb @ rotate2initial(z_min, R)
    else:  # fixed projection height
        z_min = np.array([0, 0, -plane])
        dzda = [0]
        dzdb = [0]

    # compute average coordinate for each cell, and store in 'Center' array
    mesh_rot.cell_data['Center'] = [np.sum(c.points, axis=0) / 3 for c in mesh_rot.cell]

    # compute overhang mask
    M, dMda, dMdb = smooth_overhang_connectivity(mesh, mesh_rot, R, dRda, dRdb, par)

    A, dAda, dAdb, h, dhda, dhdb = calc_V_vectorized(mesh, mesh_rot, dRda, dRdb, z_min, dzda, dzdb, par)

    volume = sum(M * A * h)
    dVda = np.sum(M * A * dhda + M * dAda * h + dMda * A * h)
    dVdb = np.sum(M * A * dhdb + M * dAdb * h + dMdb * A * h)

    return volume, [dVda, dVdb]


def SoP_connectivity_no_deriv(angles: list, mesh: pv.PolyData, par) -> float:
    plane = par['plane_offset']

    # extract angles, construct rotation matrices for x and y rotations
    _, _, R, _, _ = construct_rotation_matrix(angles[0], angles[1])

    # rotate mesh
    mesh_rot = rotate_mesh(mesh, R)

    # define z-height of projection plane for fixed projection height
    z_min = np.array([0, 0, -plane])

    # define z-height of projection plane for adaptive projection: lowest z-coordinate, closest to origin (xy norm)
    if plane <= 0:
        z_idx = np.where(mesh_rot.points[:, -1] == min(mesh_rot.points[:, -1]))[0]
        norm = np.linalg.norm(mesh_rot.points[z_idx, :-1], axis=1)
        min_norm_idx = np.argmin(norm)
        z_min = mesh_rot.points[z_idx[min_norm_idx], :] + np.array([0, 0, plane])
    else:  # fixed projection height
        z_min = np.array([0, 0, -plane])

    # compute average coordinate for each cell, and store in 'Center' array
    mesh_rot.cell_data['Center'] = [np.sum(c.points, axis=0) / 3 for c in mesh_rot.cell]

    # compute overhang mask
    M = smooth_overhang_connectivity_no_deriv(mesh, mesh_rot, R, par)

    A, h = calc_V_vect_no_deriv(mesh_rot, z_min, par)

    volume = sum(M * A * h)

    return volume


if __name__ == '__main__':

    # load file and rotate
    FILE = 'Geometries/cube_cutout.stl'

    m = pv.read(FILE)
    # m = pv.Cube()
    m = m.subdivide(2, subfilter='linear')
    m = prep_mesh(m, decimation=0)

    # set fixed projection distance
    start = time.time()

    # set parameters
    print('Generating connectivity')
    # conn = generate_connectivity_obb(m)
    conn = read_connectivity_csv('out/sim_data/connectivity2.csv')
    print(f'Connectivity took {time.time() - start} seconds')

    assert len(conn) == m.n_cells
    args = {
        'connectivity': conn,
        'build_dir': np.array([0, 0, 1]),
        'down_thresh': np.sin(np.deg2rad(0)),
        'up_thresh': np.sin(np.deg2rad(0)),
        'down_k': 10,
        'up_k': 10,
        # 'plane_offset': calc_min_projection_distance(m),
        'plane_offset': -1,
        'SoP_penalty': 1
    }

    # args['plane_offset'] = -1

    # ang = np.linspace(np.deg2rad(-45), np.deg2rad(0), 101)
    # ang = np.deg2rad([-1, 0, 1])
    # f = []
    # da = []
    # db = []
    #
    # for a in ang:
    #     f_, [da_, db_] = SoP_connectivity([a, 0], m, args)
    #     f.append(-f_)
    #     da.append(-da_)
    #     db.append(-db_)

    a = np.deg2rad(180)
    step = 201

    ang, f, da, db = grid_search_1D(SoP_connectivity, m, args, a, step, 'x')

    # ang2, f2, da2, db2 = grid_search_1D(SoP_top_smooth, m, args, a, step, 'x')
    #
    _ = plt.plot(np.rad2deg(ang), f, 'g', label='Volume')
    _ = plt.plot(np.rad2deg(ang), da, 'b.', label=r'$V_{,\alpha}$')
    _ = plt.plot(np.rad2deg(ang), db, 'k.', label=r'$V_{,\beta}$')
    _ = plt.plot(np.rad2deg(ang), finite_central_differences(f, ang), 'r.', label='Finite differences')
    plt.xlabel('Angle [deg]')
    plt.title(f'Cube with cutout - rotation about x-axis, adaptive vs fixed proj')
    _ = plt.legend()
    # plt.savefig('out/supportvolume/SoP_cube_rotx_smooth_top.svg', format='svg', bbox_inches='tight')
    plt.show()
    # #
    # ang2, f2, da2, db2 = grid_search_1D(SoP_top_cover, m, args, a, step, 'x')
    #
    # _ = plt.figure()
    # _ = plt.plot(np.rad2deg(ang), f, 'g', label='Smooth')
    # _ = plt.plot(np.rad2deg(ang), -f2, 'b', label='Original')
    # plt.xlabel('Angle [deg]')
    # plt.ylabel(fr'Volume [mm$^3$]')
    # # plt.ylim([-2, 2])
    # plt.title('Comparison of smoothing on cube with cutout')
    # plt.legend()
    # plt.savefig('out/supportvolume/SoP_cube_smooth_comp_x.svg', format='svg', bbox_inches='tight')
    # plt.show()

    end = time.time()
    print(f'Finished in {end - start} seconds')
    # make_contour_plot(np.rad2deg(ax), np.rad2deg(ay), f, 'Reference - unit cube')
    print(f'Minimum volume: {min(f)}')
