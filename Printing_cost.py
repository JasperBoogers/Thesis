from helpers.helpers import *
from sensitivities import calc_cell_sensitivities


if __name__ == '__main__':
    # load file
    mesh = decimate_quadric('Geometries/Armadillo.stl', 10000)
    # connectivity = generate_connectivity_vtk(mesh)
    connectivity = read_connectivity_csv('out/sim_data/connectivity_armadillo_10000.csv')
    mesh = prep_mesh(mesh, scaling=0.05)
    assert len(connectivity) == mesh.n_cells

    args = {
        'connectivity': connectivity,
        'build_dir': np.array([0, 0, 1]),
        'down_thresh': np.sin(np.deg2rad(0)),
        'up_thresh': np.sin(np.deg2rad(0)),
        'down_k': 10,
        'up_k': 10,
        'SoP_penalty': 0,
        'softmin_p': -15
    }

    # extract angles, construct rotation matrices for x and y rotations
    # angles = np.deg2rad([-19.40778993,  25.68747436])  # best orientation
    # angles = np.deg2rad([-84, -131])  # worst orientation
    angles = [0, 0]
    _, _, R, dRda, dRdb = construct_rotation_matrix(angles[0], angles[1])

    # rotate mesh
    mesh_rot = rotate_mesh(mesh, R)

    z_min, dz_min = mellow_min(mesh_rot.points, args['softmin_p'])
    dzda = np.sum(dz_min * np.transpose(dRda @ np.transpose(mesh.points)), axis=0)
    dzdb = np.sum(dz_min * np.transpose(dRdb @ np.transpose(mesh.points)), axis=0)

    # compute average coordinate for each cell, and store in 'Center' array
    mesh_rot.cell_data['Center'] = [np.sum(c.points, axis=0) / 3 for c in mesh_rot.cell]

    # compute overhang mask and volume per facet
    Down = smooth_heaviside(-1 * mesh_rot['Normals'][:, 2], args['down_k'], args['down_thresh'])
    Up = smooth_heaviside(mesh_rot['Normals'][:, 2], args['up_k'], args['up_thresh'])
    M, _, _ = smooth_overhang_connectivity(mesh, mesh_rot, R, dRda, dRdb, args)
    A, _, _, h, _, _ = calc_V_vectorized(mesh, mesh_rot, dRda, dRdb, z_min, dzda, dzdb, args)
    volume = M * A * h

    # define build plane
    build_plane = construct_build_plane(mesh_rot, 0)

    # extract facets with positive/negative support volume
    down_idx = np.arange(mesh_rot.n_cells)[M*A > 1e-3]
    down_facets = mesh_rot.extract_cells(down_idx).extract_surface()
    up_idx = np.arange(mesh_rot.n_cells)[M*A < -1e-3]
    up_facets = mesh_rot.extract_cells(up_idx).extract_surface()

    # extrude support volume, ie downward facets to build plane
    down_SV = []
    for i in down_idx:
        s = mesh_rot.extract_cells(i).extract_surface().extrude_trim((0, 0, -1), build_plane)
        down_SV.append(s)

    # make a gif
    p = pv.Plotter(off_screen=True)
    _ = p.add_mesh(mesh_rot, color='w', opacity=1, lighting=False, show_edges=True)
    _ = p.add_mesh(build_plane, color='k', lighting=False)
    for s in down_SV:
        p.add_mesh(s, color='b', opacity=0.5, lighting=False)
    p.camera.zoom(1.5)
    p.camera.position = (0, 10, 0)
    path = p.generate_orbital_path(n_points=100, viewup=[0, 0, 1], shift=mesh.length / 4)
    p.open_gif("out/armadillo_worst_extruded_highres.gif")
    p.show(auto_close=False)
    p.orbit_on_path(path, viewup=[0, 0, 1], write_frames=True)
    p.close()

    # plot support requirement of mesh in given orientation
    mesh = calc_cell_sensitivities(mesh, angles, args)
    p = pv.Plotter()
    _ = p.add_mesh(mesh, lighting=True, scalars='MA', cmap='RdYlBu', clim=[-1, 1], show_edges=False,
                   scalar_bar_args={"title": "Support requirement",
                                    "position_x": 0.05,
                                    "position_y": 0.9})
    _ = p.add_mesh(construct_build_plane(mesh, 0), lighting=True, color='k', opacity=0.8)
    p.show()
