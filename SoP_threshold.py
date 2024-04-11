import csv
import time
import pyvista as pv
import numpy as np
from helpers import *
from SoP import SoP_top_cover
from scipy.optimize import minimize


def SoP_smooth(angles: list, mesh: pv.PolyData, par) -> tuple[float, list]:
    plane = par['plane_offset']

    # extract angles, construct rotation matrices for x and y rotations
    Ra, Rb, R, dRda, dRdb = construct_rotation_matrix(angles[0], angles[1])

    # rotate mesh
    mesh_rot = rotate_mesh(mesh, R)

    # define z-height of projection plane for fixed projection height
    z_min = np.array([0, 0, -plane])
    dzda = dzdb = [0]

    # uncomment for adaptive projection height
    # z_min = mesh_rot.points[np.argmin(mesh_rot.points[:, -1]), :]
    # dzda = dRda @ rotate2initial(z_min, R)
    # dzdb = dRdb @ rotate2initial(z_min, R)

    # compute average coordinate for each cell, and store in 'Center' array
    mesh_rot.cell_data['Center'] = [np.sum(c.points, axis=0) / 3 for c in mesh_rot.cell]

    # compute overhang mask
    M, dMda, dMdb = smooth_overhang_connectivity(mesh, mesh_rot, R, dRda, dRdb, par)

    A, dAda, dAdb, h, dhda, dhdb = calc_V_vectorized(mesh, mesh_rot, dRda, dRdb, z_min, dzda, dzdb, par)

    volume = sum(M * A * h)
    dVda = np.sum(M * A * dhda + M * dAda * h + dMda * A * h)
    dVdb = np.sum(M * A * dhdb + M * dAdb * h + dMdb * A * h)

    return volume, [dVda, dVdb]


def overhang_mask_gif(mesh, filename):
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


if __name__ == '__main__':
    start = time.time()

    # load file
    FILE = 'Geometries/cube_cutout.stl'
    m = pv.read(FILE)
    m = m.subdivide(2, subfilter='linear')
    m = prep_mesh(m, decimation=0, translate=True)

    # overhang_mask_gif(m, 'out/supportvolume/SoP/OverhangMask4_averaged.gif')

    # set parameters
    thresh = 0
    OVERHANG_THRESHOLD = np.sin(np.deg2rad(thresh))  # angle between build plane and facet normal
    PLANE_OFFSET = calc_min_projection_distance(m)
    print('Generating connectivity')
    # conn = generate_connectivity_obb(m)
    conn = read_connectivity_csv('out/sim_data/connectivity2.csv')
    print(f'Connectivity took {time.time() - start} seconds')

    assert len(conn) == m.n_cells

    args = {
        'connectivity': conn,
        'build_dir': np.array([0, 0, 1]),
        'down_thresh': OVERHANG_THRESHOLD,
        'up_thresh': 0,
        'down_k': 10,
        'up_k': 10,
        'plane_offset': PLANE_OFFSET,

    }
    # ang = np.deg2rad([0, -90, -40])
    # f = []
    # da = []
    # db = []
    #
    # for a in ang:
    #     f_, [da_, db_] = SoP_smooth([a, 0], m, args)
    #     f.append(f_)
    #     da.append(da_)
    #     db.append(db_)
    #
    a = np.deg2rad(180)
    step = 41

    # ax, ay, f, da, db = grid_search(SoP_smooth, m, args, a, step)
    #
    # with open('out/sim_data/bunny_contour_f_45deg.csv', 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerows(f)
    #
    # with open('out/sim_data/bunny_contour_dfda_45deg.csv', 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerows(da)
    #
    # with open('out/sim_data/bunny_contour_dfdb_45deg.csv', 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerows(db)

    # ax = ay = np.linspace(-np.pi, np.pi, step)
    # f = read_csv('out/sim_data/bunny_contour_dfdb.csv')
    #
    # make_contour_plot(np.rad2deg(ax), np.rad2deg(ay), f, f'Stanford Bunny support volume, derivative about y-axis',
    #                   'out/contourplot/Bunny/contourplot_bunny_dfdb.svg')

    step = 201
    axis='x'
    ang, f, da, db = grid_search_1D(SoP_smooth, m, args, a, step, axis)

    _ = plt.plot(np.rad2deg(ang), f, 'g', label='Volume')
    _ = plt.plot(np.rad2deg(ang), da, 'b.', label=r'$V_{,\alpha}$')
    _ = plt.plot(np.rad2deg(ang), db, 'k.', label=r'$V_{,\beta}$')
    _ = plt.plot(np.rad2deg(ang), finite_central_differences(f, ang), 'r.', label='Finite differences')
    plt.xlabel('Angle [deg]')
    plt.title(f'Cube, overhang threshold=0 deg, rotation about {axis}-axis')
    _ = plt.legend()
    # plt.savefig(f'out/supportvolume/SoP/SoP_chair_rot{axis}_0deg.svg', format='svg', bbox_inches='tight')
    plt.show()



    # m2 = m.subdivide(2, subfilter='linear')
    # ang2, f2, da2, db2 = grid_search_1D(SoP_smooth, m2, OVERHANG_THRESHOLD, PLANE_OFFSET, a, step, 'x')
    #
    # m3 = m.subdivide(3, subfilter='linear')
    # ang3, f3, da3, db3 = grid_search_1D(SoP_smooth, m3, OVERHANG_THRESHOLD, PLANE_OFFSET, a, step, 'x')
    #
    # _ = plt.figure()
    # _ = plt.plot(np.rad2deg(ang), f, label=f'{m.n_cells} facets')
    # _ = plt.plot(np.rad2deg(ang), -f2, label=f'{m2.n_cells} facets')
    # _ = plt.plot(np.rad2deg(ang), -f3, label=f'{m3.n_cells} facets')
    #
    # plt.xlabel('Angle [deg]')
    # plt.ylabel(fr'Volume [mm$^3$]')
    # # plt.ylim([-2, 2])
    # plt.title('Effect of mesh size on support volume of a cube with cutout')
    # plt.legend()
    # plt.savefig('out/supportvolume/SoP/SoP_cube_mesh_comp_x.svg', format='svg', bbox_inches='tight')
    # plt.show()
    #
    # _ = plt.figure()
    # _ = plt.plot(np.rad2deg(ang), da, label=f'{m.n_cells} facets')
    # _ = plt.plot(np.rad2deg(ang), -da2, label=f'{m2.n_cells} facets')
    # _ = plt.plot(np.rad2deg(ang), -da3, label=f'{m3.n_cells} facets')
    #
    # plt.xlabel('Angle [deg]')
    # plt.ylabel(fr'Volume [mm$^3$]')
    # # plt.ylim([-2, 2])
    # plt.title('Effect of mesh size on support volume derivative')
    # plt.legend()
    # plt.savefig('out/supportvolume/SoP/SoP_cube_mesh_x_derivative_comp.svg', format='svg', bbox_inches='tight')
    # plt.show()

    # f = read_csv('out/sim_data/cube_cutout_contour_f.csv')
    # ax = ay = np.linspace(-np.pi, np.pi, 21)
    # x0 = extract_x0(ax, ay, f, 5)
    #
    # res = []
    #
    # for i in range(5):
    #     s = time.time()
    #
    #     # set initial condition
    #     a = np.array(x0[i])
    #     print(f'Iteration {i + 1} with x0: {np.rad2deg(a)} degrees')
    #
    #     y = minimize(SoP_smooth, a, jac=True, args=(m, args))
    #     end = time.time() - s
    #     print(y)
    #     print(f'Optimal orientation at {np.rad2deg(y.x)} degrees')
    #     print(f'Computation time: {end} s')
    #     res.append(y)

    end = time.time()
    print(f'Finished in {end - start} seconds')
