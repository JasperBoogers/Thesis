import time
import pyvista as pv
import numpy as np
from helpers import *
from SoP import SoP_top_cover


def SoP_smooth(angles: list, mesh: pv.PolyData, threshold: float, plane: float) -> tuple[float, list]:
    # extract angles, construct rotation matrices for x and y rotations
    Ra, Rb, R, dRda, dRdb = construct_rotation_matrix(angles[0], angles[1])

    # rotate mesh
    mesh_rot = rotate_mesh(mesh, R)

    # define z-height of projection plane for fixed projection height
    build_dir = np.array([0, 0, -1])
    z_min = np.array([0, 0, -plane])
    dzda = dzdb = [0]

    # compute average coordinate for each cell, and store in 'Center' array
    mesh_rot.cell_data['Center'] = [np.sum(c.points, axis=0) / 3 for c in mesh_rot.cell]

    # compute overhang mask
    k = 10
    overhang = smooth_overhang(mesh_rot, build_dir, threshold, k)

    # calculate volume
    volume = 0.0
    dVda = 0.0
    dVdb = 0.0
    for idx in range(mesh_rot.n_cells):
        # extract points and normal vector from cell
        cell = mesh_rot.extract_cells(idx)
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
        H = overhang[idx]
        dHda = H * (1 - H) * -2 * k * dnda[-1]
        dHdb = H * (1 - H) * -2 * k * dndb[-1]

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

    # load file
    FILE = 'Geometries/chair.stl'
    m = pv.read(FILE)
    m = prep_mesh(m, decimation=0)
    m = m.subdivide(2, subfilter='linear')

    # set parameters
    OVERHANG_THRESHOLD = -1e-5
    PLANE_OFFSET = calc_min_projection_distance(m)

    # grid search
    start = time.time()

    # angles = np.deg2rad([-41, -40, -39])
    # f = []
    # da = []
    # db = []
    #
    # for a in angles:
    #     f_, [da_, db_] = SoP_smooth([a, 0], m, OVERHANG_THRESHOLD, PLANE_OFFSET)
    #     f.append(-f_)
    #     da.append(-da_)
    #     db.append(-db_)

    a = np.deg2rad(180)
    step = 201
    ang, f, da, db = grid_search_1D(SoP_smooth, m, OVERHANG_THRESHOLD, PLANE_OFFSET, a, step, 'y')
    f = -f
    da = -da
    db = -db

    _ = plt.plot(np.rad2deg(ang), f, 'g', label='Volume')
    _ = plt.plot(np.rad2deg(ang), da, 'b.', label=r'$V_{,\alpha}$')
    _ = plt.plot(np.rad2deg(ang), db, 'k.', label=r'$V_{,\beta}$')
    _ = plt.plot(np.rad2deg(ang)[:-1], finite_forward_differences(f, ang), 'r.', label='Finite differences')
    plt.xlabel('Angle [deg]')
    # plt.ylim([-2, 2])
    plt.title(f'Cube - rotation about y-axis, smoothened')
    _ = plt.legend()
    # plt.savefig('out/supportvolume/SoP/SoP_chair_roty_smooth.svg', format='svg', bbox_inches='tight')
    plt.show()

    # ang2, f2, da2, db2 = grid_search_1D(SoP_top_cover, m, OVERHANG_THRESHOLD, PLANE_OFFSET, a, step, 'y')
    #
    # _ = plt.figure()
    # _ = plt.plot(np.rad2deg(ang), f, 'g', label='Smooth')
    # _ = plt.plot(np.rad2deg(ang), -f2, 'b', label='Original')
    # plt.xlabel('Angle [deg]')
    # plt.ylabel(fr'Volume [mm$^3$]')
    # # plt.ylim([-2, 2])
    # plt.title('Comparison of smoothing on a chair')
    # plt.legend()
    # plt.savefig('out/supportvolume/SoP/SoP_chair_smooth_comp_y.svg', format='svg', bbox_inches='tight')
    # plt.show()


    end = time.time()
    print(f'Finished in {end - start} seconds')