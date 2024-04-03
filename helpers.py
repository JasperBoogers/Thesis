import pyvista as pv
import numpy as np
from scipy.spatial.transform import Rotation
from matplotlib import pyplot as plt
from os import cpu_count
from joblib import delayed, Parallel
from math_helpers import *


def prep_mesh(mesh: pv.PolyData | pv.DataSet, decimation=0.9, flip=False, translate=True) -> pv.PolyData:
    # ensure mesh is only triangles
    mesh.triangulate(inplace=True)

    # decimate mesh by decimate*100%
    mesh.decimate_pro(decimation)

    # (re)compute normals, and flip normal direction if needed
    mesh.compute_normals(inplace=True, flip_normals=flip)

    # move mesh center of mass to origin
    if translate:
        mesh = mesh.translate(-mesh.center_of_mass(), inplace=False)

    return mesh


def rotate_mesh(m: pv.PolyData | pv.DataSet, rot: np.ndarray | Rotation) -> pv.DataSet:
    # Rotate the mesh through the rotation obj R
    tfm = np.identity(4)

    if isinstance(rot, np.ndarray):
        tfm[:-1, :-1] = rot
    else:
        tfm[:-1, :-1] = rot.as_matrix()
    return m.transform(tfm, inplace=False)


def extract_overhang(m: pv.PolyData | pv.DataSet, t: float) -> pv.PolyData:
    idx = np.arange(m.n_cells)[m['Normals'][:, 2] < t]
    overhang = m.extract_cells(idx)
    return overhang.extract_surface()


def construct_build_plane(m: pv.PolyData | pv.DataSet, offset: float) -> pv.PolyData:
    bounds = m.bounds
    return pv.Plane(center=(0, 0, bounds[-2] - offset),
                    i_size=1.1 * (bounds[1] - bounds[0]),
                    j_size=1.1 * (bounds[3] - bounds[2]),
                    direction=(0, 0, 1))


def construct_supports(o: pv.PolyData | pv.DataSet, p: pv.PolyData) -> pv.PolyData:
    SV = o.extrude_trim((0, 0, -1), p)
    SV.triangulate(inplace=True)
    SV.compute_normals(inplace=True, flip_normals=True)
    return SV


def construct_support_volume(mesh: pv.PolyData | pv.DataSet, threshold: float, plane_offset: float = 1.0) -> tuple[
    pv.PolyData, pv.PolyData, pv.PolyData]:
    # extract overhanging surfaces
    overhang = extract_overhang(mesh, threshold)

    # construct print bed plane based on lowest mesh point,
    # add an offset to ensure proper triangulation
    plane = construct_build_plane(mesh, plane_offset)

    # extrude overhanging surfaces to projection plane
    SV = construct_supports(overhang, plane)

    return overhang, plane, SV


def make_surface_plot(x: np.ndarray, y: np.ndarray, f: np.ndarray):
    x, y = np.meshgrid(np.rad2deg(x), np.rad2deg(y))
    surf = pv.StructuredGrid(x, y, f)
    surf_plot = pv.Plotter()
    surf_plot.add_mesh(surf, scalars=surf.points[:, -1], show_edges=True,
                       scalar_bar_args={'vertical': True})
    surf_plot.set_scale(zscale=5)
    surf_plot.show_grid()
    surf_plot.show()


def make_contour_plot(x: np.ndarray, y: np.ndarray, f: np.ndarray, titel=None, save=None):
    x, y = np.meshgrid(x, y)
    fig = plt.figure()
    cp = plt.contourf(x, y, f)
    fig.colorbar(cp)
    plt.xlabel('Rotation about x-axis [deg]')
    plt.ylabel('Rotation about y-axis [deg]')
    if titel is not None:
        plt.title(titel)
    if save is not None:
        plt.savefig(save, format='svg', bbox_inches='tight')
    plt.show()


def calc_min_projection_distance(m: pv.PolyData | pv.DataSet) -> float:
    bounds = m.bounds
    x = bounds[1] - bounds[0]
    y = bounds[3] - bounds[2]
    z = bounds[5] - bounds[4]
    return np.linalg.norm([x, y, z]) / 2


def extract_top_cover(m, upward):
    # set bounds
    bounds = m.bounds
    z_min = bounds[-2] - 5
    z_max = bounds[-1] + 5

    top = set()
    not_top = set()
    lines = []

    for i in upward:

        # continue if already checked
        if i in top:
            continue
        if i in not_top:
            continue

        # extract cell
        cell = m.extract_cells(i)

        # center is average of point coordinates
        center = cell['Center']
        center = center[0]

        # generate line through center and extract intersecting cells
        line = pv.Line([center[0], center[1], z_min], [center[0], center[1], z_max])
        lines.append(line)

        # check if any cells intersect that line
        intersect = m.find_cells_intersecting_line(line.points[0], line.points[1])

        if len(intersect) > 1:
            # centers = m.extract_cells(intersect)['Center']
            # max_idx = np.argwhere(centers[:, -1] == np.max(centers[:, -1])).flatten().tolist()
            max_idx = -1

            # add cell index with max z to top
            top.add(intersect[max_idx])

            # add other cells to not_top
            not_top.update(np.delete(intersect, max_idx))
        elif len(intersect) > 0:
            # only one intersecting cell -> top cover
            top.update(intersect)
        else:
            pass

    return list(top), lines


def calc_V_under_triangle(cell, angles, build_dir, z_min):
    # extract angles, construct rotation matrices for x and y rotations
    Rx, Ry, R, dRdx, dRdy = construct_rotation_matrix(angles[0], angles[1])

    # extract points and normals
    points = np.transpose(cell.points)
    normal = np.transpose(cell.cell_data.active_normals)

    # normal has to be of unit length for area calculation
    normal /= np.linalg.norm(normal)

    # compute initial points and normal vector
    points0 = rotate2initial(points, R)
    normal0 = rotate2initial(normal, R)

    # calculate area and height
    A = cell.area * build_dir.dot(normal)[0]
    h = sum(points[-1]) / 3 - z_min[-1]
    V = A * h

    # calculate area derivative
    dAdx = cell.area * build_dir.dot(dRdx @ normal0)[0]
    dAdy = cell.area * build_dir.dot(dRdy @ normal0)[0]

    # calculate height derivative
    dhdx = sum((dRdx @ points0)[-1]) / 3
    dhdy = sum((dRdy @ points0)[-1]) / 3

    # calculate volume derivative and sum
    dVdx = A * dhdx + h * dAdx
    dVdy = A * dhdy + h * dAdy

    return V, dVdx, dVdy


def extract_correction_idx(mesh, overhang_idx):
    # set bounds
    bounds = mesh.bounds
    z_min = bounds[-2] - 5
    z_max = bounds[-1] + 5

    correction_idx = set()
    lines = []

    for i in overhang_idx:

        # extract cell
        cell = mesh.extract_cells(i)

        # center is average of point coordinates
        center = cell['Center']
        center = center[0]

        # generate line below center and extract intersecting cells
        line = pv.Line([center[0], center[1], z_min], center)
        lines.append(line)

        # check if any cells intersect that line
        intersect = mesh.find_cells_intersecting_line(line.points[0], line.points[1])

        # do not consider last index, as that is the overhanging cell
        if i in intersect:
            intersect = intersect[:-1]

        if len(intersect) > 0:
            centers = np.array([mesh.extract_cells(c)['Center'][0] for c in intersect])
            max_idx = np.argwhere(centers[:, -1] == np.max(centers[:, -1])).flatten().tolist()

            # add cell index with max z to correction
            correction_idx.update(intersect[max_idx])

    return list(correction_idx), lines


def grid_search(fun, mesh, overhang, offset, max_angle, angle_step):
    # grid search parameters
    ax = ay = np.linspace(-max_angle, max_angle, angle_step)
    # f = np.zeros((ax.shape[0], ay.shape[0]))
    #
    # for i, x in enumerate(ax):
    #     for j, y in enumerate(ay):
    #         f[j, i] = fun([x, y], mesh, overhang, offset)[0]

    f = Parallel(n_jobs=cpu_count())(delayed(fun)([x, y], mesh, overhang, offset) for x in ax for y in ay)
    f, _, = zip(*f)
    f = np.reshape(f, (len(ax), len(ay)))
    f = np.transpose(f)
    return ax, ay, f


def grid_search_1D(fun, mesh, overhang, offset, max_angle, angle_step, dim='x'):
    a = np.linspace(-max_angle, max_angle, angle_step)

    if dim == 'x':
        f = Parallel(n_jobs=cpu_count())(delayed(fun)([ai, 0], mesh, overhang, offset) for ai in a)
    elif dim == 'y':
        f = Parallel(n_jobs=cpu_count())(delayed(fun)([0, ai], mesh, overhang, offset) for ai in a)
    else:
        print('Invalid')
        return

    f, d = zip(*f)
    da, db = zip(*d)
    return np.array(a), np.array(f), np.array(da), np.array(db)


def extract_x0(ax, ay, f, n):
    flat_idx = np.argpartition(f.ravel(), -n)[-n:]
    row_idx, col_idx = np.unravel_index(flat_idx, f.shape)
    return [[ax[row_idx[k]], ay[col_idx[k]]] for k in range(n)]


def calc_cell_sensitivities(mesh: pv.PolyData | pv.DataSet, angles: list | np.ndarray,
                            build_dir: list | np.ndarray, z_min: list | np.ndarray):
    _, _, R, _, _ = construct_rotation_matrix(angles[0], angles[1])
    mesh = rotate_mesh(mesh, R)

    # compute cell areas
    mesh = mesh.compute_cell_sizes(length=False, volume=False)

    # compute sensitivities for all cells
    res = Parallel(n_jobs=cpu_count())(
        delayed(calc_V_under_triangle)(mesh.extract_cells(i), angles, build_dir, z_min) for i in range(mesh.n_cells))

    f, dx, dy = zip(*res)

    # f = []
    # dx = []
    # dy = []
    #
    # for i in range(mesh.n_cells):
    #     c = mesh.extract_cells(i)
    #
    #     f_, dx_, dy_ = calc_V_under_triangle(c, angles, build_dir, z_min)
    #     f.append(f_)
    #     dx.append(dx_)
    #     dy.append(dy_)

    thresh = mesh['Normals'][:, 2] < -1e-6

    mesh.cell_data['dVda'] = np.array(dx) / mesh.cell_data['Area'] * thresh
    mesh.cell_data['dVdb'] = np.array(dy) / mesh.cell_data['Area'] * thresh
    mesh.cell_data['dV'] = np.linalg.norm(np.array([dx, dy]), axis=0) / mesh.cell_data['Area'] * thresh
    mesh.cell_data['V'] = np.array(f) / mesh.cell_data['Area'] * thresh
    return mesh


def plot_cell_sensitivities(mesh: pv.PolyData | pv.DataSet, axis: str = 'x') -> None:
    p = pv.Plotter()
    if axis == 'x':
        scalars = 'dVda'
    elif axis == 'y':
        scalars = 'dVdb'
    elif axis == 'V':
        scalars = 'V'
    else:
        scalars = 'dV'

    _ = p.add_mesh(mesh, lighting=False, scalars=scalars, cmap='RdYlGn', show_edges=True)
    p.add_axes()
    p.show()


def build_top_mask(mesh, upward_ids, top_ids):
    # mask = np.zeros(mesh.n_cells)
    mask = smooth_heaviside(mesh['Normals'][:, 2], 10, 1e-5)

    for i in upward_ids:

        # get center
        cell = mesh.extract_cells(i)
        center = cell['Center'][0]
        area = cell.area

        # get (upward facing) neighbours
        neighbors = set()
        for n in mesh.cell_neighbors(i, 'points'):
            for j in mesh.cell_neighbors(n, 'points'):
                if j in upward_ids:
                    neighbors.add(j)

        # neighbors = [n for n in mesh.cell_neighbors(i, 'points') if n in upward_ids]
        # neighbors = [n for n in mesh.cell_neighbors(j, 'points') for j in neighbors if n in upward_ids]
        neighbors = list(neighbors)
        # neighbors = mesh.cell_neighbors(i, 'points')
        neighbors.append(i)

        val = 0
        dist = 0
        for c in neighbors:
            cell = mesh.extract_cells(c)
            d = 1 - (np.linalg.norm(cell['Center'][0] - center))

            if c in top_ids:
                v = 1
            else:
                v = 0

            val += v * d
            dist += d

        mask[i] = val/dist

    return mask


def smooth_top_cover(mesh, upward, build_dir):
    mask = smooth_heaviside(mesh['Normals'][:, 2], 10, 1e-5)
    overhang = smooth_heaviside(mesh['Normals'][:, 2], 10, 0.5)

    # create rotation matrices for projection rays
    proj_angle = np.deg2rad(5)
    Rx, Ry, _, _, _ = construct_rotation_matrix(proj_angle, proj_angle)

    # rotate all rays by 45 deg around z-axis
    a = np.deg2rad(45)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])

    n_rays = 5
    directs = [build_dir, Rx @ build_dir, np.transpose(Rx) @ build_dir, Ry @ build_dir, np.transpose(Ry) @ build_dir]
    directs = np.transpose(Rz @ np.transpose(directs))

    for idx in range(mesh.n_cells):

        # get cell and normal
        cell = mesh.extract_cells(idx)

        # center is average of point coordinates
        center = cell['Center']
        center = center[0]

        # check if any cells intersect that line
        origins = [center] * n_rays
        _, inter_rays, inter_cells = mesh.multi_ray_trace(origins, directs, first_point=False)

        # drop intersections with cell idx
        # inter_points = inter_points[inter_cells != idx]
        inter_rays = inter_rays[inter_cells != idx]
        inter_cells = inter_cells[inter_cells != idx]

        for r in set(inter_rays):
            c = mesh.extract_cells(inter_cells[inter_rays == r][0])['Center']
            mask[idx] -= np.dot(build_dir, c[0] - center)**2 / n_rays * overhang[idx]
    return mask


def smooth_overhang(mesh, rotated_mesh: pv.PolyData | pv.DataSet, R, dRda, dRdb, cast_dir: np.ndarray,
                    threshold: float, smoothing: int | float) -> tuple:

    # construct upward, downward and combined fields
    k = smoothing
    Down = smooth_heaviside(-1 * rotated_mesh['Normals'][:, 2], k, -threshold)
    Up = smooth_heaviside(rotated_mesh['Normals'][:, 2], k, 0)
    mask = np.zeros_like(Down)
    mask_avg = np.zeros_like(Down)

    # set up derivative fields
    dmask_da = np.zeros_like(Down)
    dmask_db = np.zeros_like(Down)
    dmask_da_avg = np.zeros_like(Down)
    dmask_db_avg = np.zeros_like(Down)

    # create rotation matrices for projection rays
    angle = np.deg2rad(10)
    Rx, Ry, _, _, _ = construct_rotation_matrix(angle, angle)

    # rotate all rays by 45 deg around z-axis
    Rz = np.array([[np.cos(np.deg2rad(45)), -np.sin(np.deg2rad(45)), 0],
                   [np.sin(np.deg2rad(45)), np.cos(np.deg2rad(45)), 0], [0, 0, 1]])

    # construct direction vectors for ray tracing
    directs = np.array([cast_dir, Rx @ cast_dir, np.transpose(Rx) @ cast_dir, Ry @ cast_dir,
               np.transpose(Ry) @ cast_dir])
    # directs = np.append(directs, np.array([np.transpose(Rz @ np.transpose(i)) for i in directs[1:,:]]), axis=0)
    directs = np.transpose(Rz @ np.transpose(directs))
    n_rays = directs.shape[0]

    # loop over cells and subtract value to compensate for overhangs
    for idx in range(rotated_mesh.n_cells):
        # extract points and normal vector from cell
        cell = rotated_mesh.extract_cells(idx)

        normals0 = np.transpose(mesh['Normals'][idx])
        dnda = dRda @ normals0
        dndb = dRdb @ normals0

        D = Down[idx]
        dDown_da = D * (1 - D) * 2 * k * -dnda[-1]
        dDown_db = D * (1 - D) * 2 * k * -dndb[-1]

        mask[idx] += D
        dmask_da[idx] += dDown_da
        dmask_db[idx] += dDown_db

        # setup center
        center = cell['Center'][0]
        for d in directs:
            _, ids = rotated_mesh.ray_trace(center, center + 100 * d)

            # select first cell that is not idx
            i = ids[ids != idx]
            if len(i) > 0:
                j = i[0]

                c = rotated_mesh.extract_cells(j)['Center'][0]
                l = c - center
                l /= np.linalg.norm(l)

                U = Up[j]
                mask_val = np.dot(cast_dir, l) / n_rays * U
                mask[i[0]] += mask_val/2

                normals0 = np.transpose(mesh['Normals'][j])
                dnj_da = dRda @ normals0
                dnj_db = dRdb @ normals0

                dUp_da = U * (1 - U) * 2 * k * dnj_da[-1]
                dUp_db = U * (1 - U) * 2 * k * dnj_db[-1]

                dl_da = dRda @ rotate2initial(l, R)
                dl_db = dRdb @ rotate2initial(l, R)

                dmask_da[j] += (np.dot(cast_dir, dl_da) / n_rays * U + np.dot(cast_dir, l) / n_rays * dUp_da)/2
                dmask_db[j] += (np.dot(cast_dir, dl_db) / n_rays * U + np.dot(cast_dir, l) / n_rays * dUp_db)/2

    # for idx in range(rotated_mesh.n_cells):
    #     neighbours = rotated_mesh.cell_neighbors(idx, 'edges')
    #
    #     val = 0
    #     d = 0
    #     for n in neighbours:
    #         val += np.dot(rotated_mesh['Normals'][idx], rotated_mesh['Normals'][n]) * mask[n]
    #         d += np.dot(rotated_mesh['Normals'][idx], rotated_mesh['Normals'][n])
    #     mask_avg[idx] = (val + 2 * mask[idx]) / (d + 2)
    #
        # neighbours = rotated_mesh.cell_neighbors(idx, 'edges')
        # mask_avg[idx] = (np.sum(mask[neighbours]) + 1 * mask[idx]) / (len(neighbours) + 1)
    #     dmask_da_avg[idx] = (np.sum(dmask_da[neighbours]) + 1 * dmask_da[idx]) / (len(neighbours) + 1)
    #     dmask_db_avg[idx] = (np.sum(dmask_db[neighbours]) + 1 * dmask_db[idx]) / (len(neighbours) + 1)

    return mask, dmask_da, dmask_db
