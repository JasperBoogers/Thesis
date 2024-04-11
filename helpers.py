import pyvista as pv
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from matplotlib import pyplot as plt
from os import cpu_count
from vtk import vtkOBBTree, vtkPoints, vtkCellData, vtkCellArray, vtkPolyData, vtkIdList
from joblib import delayed, Parallel
from math_helpers import *
from io_helpers import *


def prep_mesh(mesh: pv.PolyData | pv.DataSet, decimation=0, flip=False, translate=True) -> pv.DataSet:
    # set to double precision
    mesh = mesh.points_to_double()

    # ensure mesh is only triangles
    mesh.triangulate(inplace=True)

    # decimate mesh by decimate*100%
    mesh = mesh.decimate_pro(decimation)

    # (re)compute normals, and flip normal direction if needed
    mesh.compute_normals(inplace=True, flip_normals=flip)

    # compute cell areas
    mesh = mesh.compute_cell_sizes()

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


def calc_V_under_triangle(cell, angles, z_min, par):
    build_dir = par['build_dir']

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


def calc_V_vectorized(mesh, mesh_rot, dRdx, dRdy, z_min, dzdx, dzdy, par):
    build_dir = par['build_dir']

    # extract points and normals
    points = np.array([c.points for c in mesh_rot.cell])
    points0 = np.array([c.points for c in mesh.cell])
    normals = mesh_rot['Normals']
    normals0 = mesh['Normals']

    # derivative of normals
    dnda = np.transpose(dRdx @ np.transpose(normals0))
    dndb = np.transpose(dRdy @ np.transpose(normals0))

    # compute area and derivative
    area = mesh['Area'] * np.dot(-build_dir, np.transpose(normals))
    dAdx = mesh['Area'] * np.dot(-build_dir, np.transpose(dnda))
    dAdy = mesh['Area'] * np.dot(-build_dir, np.transpose(dndb))

    # compute height and derivative
    height = np.sum(points[:, :, -1], axis=1) / 3 - z_min[-1]
    dhdx = np.sum(np.transpose(dRdx @ np.transpose(points0, (0, 2, 1)), (0, 2, 1))[:, :, -1], axis=1) / 3 - dzdx[-1]
    dhdy = np.sum(np.transpose(dRdy @ np.transpose(points0, (0, 2, 1)), (0, 2, 1))[:, :, -1], axis=1) / 3 - dzdy[-1]

    return area, dAdx, dAdy, height, dhdx, dhdy


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


def grid_search(fun, mesh, args, max_angle, angle_step):
    # grid search parameters
    ax = ay = np.linspace(-max_angle, max_angle, angle_step)
    # f = np.zeros((ax.shape[0], ay.shape[0]))
    #
    # for i, x in enumerate(ax):
    #     for j, y in enumerate(ay):
    #         f[j, i] = fun([x, y], mesh, overhang, offset)[0]

    f = Parallel(n_jobs=cpu_count())(delayed(fun)([x, y], mesh, args) for x in ax for y in ay)
    f, d = zip(*f)
    dx, dy = zip(*d)

    f = np.reshape(f, (len(ax), len(ay)))
    f = np.transpose(f)

    dx = np.reshape(dx, (len(ax), len(ay)))
    dx = np.transpose(dx)
    dy = np.reshape(dy, (len(ax), len(ay)))
    dy = np.transpose(dy)

    return ax, ay, f, dx, dy


def grid_search_1D(fun, mesh, func_args, max_angle, angle_step, dim='x'):
    a = np.linspace(-max_angle, max_angle, angle_step)

    if dim == 'x':
        f = Parallel(n_jobs=cpu_count())(delayed(fun)([ai, 0], mesh, func_args) for ai in a)
    elif dim == 'y':
        f = Parallel(n_jobs=cpu_count())(delayed(fun)([0, ai], mesh, func_args) for ai in a)
    else:
        print('Invalid')
        return

    f, d = zip(*f)
    da, db = zip(*d)
    return np.array(a), np.array(f), np.array(da), np.array(db)


def extract_x0(ax, ay, f, n, smallest=True):
    if smallest:
        flat_idx = np.argpartition(f.ravel(), n)[:n]
    else:
        flat_idx = np.argpartition(f.ravel(), -n)[-n:]
    row_idx, col_idx = np.unravel_index(flat_idx, f.shape)
    return [[ax[row_idx[k]], ay[col_idx[k]]] for k in range(n)]


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

        mask[i] = val / dist

    return mask


def smooth_top_cover(mesh, par):
    build_dir = par['build_dir']
    down_thresh = par['down_thresh']
    up_thresh = par['up_thresh']
    down_k = par['down_k']
    up_k = par['up_k']

    mask = smooth_heaviside(mesh['Normals'][:, 2], down_k, down_thresh)
    overhang = smooth_heaviside(mesh['Normals'][:, 2], up_k, up_thresh)

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
            mask[idx] -= np.dot(build_dir, c[0] - center) ** 2 / n_rays * overhang[idx]
    return mask


def smooth_overhang(mesh, rotated_mesh: pv.PolyData | pv.DataSet, R, dRda, dRdb, par: dict) -> tuple:
    build_dir = par['build_dir']
    down_thresh = par['down_thresh']
    up_thresh = par['up_thresh']
    down_k = par['down_k']
    up_k = par['up_k']

    cast_dir = -build_dir

    # construct upward, downward and combined fields
    Down = smooth_heaviside(-1 * rotated_mesh['Normals'][:, 2], down_k, down_thresh)
    Up = smooth_heaviside(rotated_mesh['Normals'][:, 2], up_k, up_thresh)
    mask = np.zeros_like(Down)

    # set up derivative fields
    dmask_da = np.zeros_like(Down)
    dmask_db = np.zeros_like(Down)

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
        dDown_da = D * (1 - D) * 2 * down_k * -dnda[-1]
        dDown_db = D * (1 - D) * 2 * down_k * -dndb[-1]

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
                mask[i[0]] += mask_val / 2

                normals0 = np.transpose(mesh['Normals'][j])
                dnj_da = dRda @ normals0
                dnj_db = dRdb @ normals0

                dUp_da = U * (1 - U) * 2 * up_k * dnj_da[-1]
                dUp_db = U * (1 - U) * 2 * up_k * dnj_db[-1]

                dl_da = dRda @ rotate2initial(l, R)
                dl_db = dRdb @ rotate2initial(l, R)

                dmask_da[j] += (np.dot(cast_dir, dl_da) / n_rays * U + np.dot(cast_dir, l) / n_rays * dUp_da) / 2
                dmask_db[j] += (np.dot(cast_dir, dl_db) / n_rays * U + np.dot(cast_dir, l) / n_rays * dUp_db) / 2

    return mask, dmask_da, dmask_db


def smooth_overhang_upward(mesh, rotated_mesh: pv.PolyData | pv.DataSet, R, dRda, dRdb, par) -> tuple:
    build_dir = par['build_dir']
    down_thresh = par['down_thresh']
    up_thresh = par['up_thresh']
    down_k = par['down_k']
    up_k = par['up_k']

    # construct upward, downward and combined fields
    Down = smooth_heaviside(-1 * rotated_mesh['Normals'][:, 2], down_k, down_thresh)
    Up = smooth_heaviside(rotated_mesh['Normals'][:, 2], up_k, up_thresh)
    mask = np.zeros_like(Down)

    # set up derivative fields
    dmask_da = np.zeros_like(Down)
    dmask_db = np.zeros_like(Down)

    # create rotation matrices for projection rays
    angle = np.deg2rad(30)
    Rx, Ry, _, _, _ = construct_rotation_matrix(angle, angle)

    # rotate all rays by 45 deg around z-axis
    Rz = np.array([[np.cos(np.deg2rad(45)), -np.sin(np.deg2rad(45)), 0],
                   [np.sin(np.deg2rad(45)), np.cos(np.deg2rad(45)), 0], [0, 0, 1]])

    # hard-code cast dir
    cast_dir = build_dir

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
        dDown_da = D * (1 - D) * 2 * down_k * -dnda[-1]
        dDown_db = D * (1 - D) * 2 * down_k * -dndb[-1]

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

                Dj = Down[j]
                Ui = Up[idx]
                v = 3
                mask_val = np.dot(cast_dir, l) / n_rays * Dj * Ui
                mask[idx] += mask_val / v

                normals0 = np.transpose(mesh['Normals'][j])
                dnj_da = dRda @ normals0
                dnj_db = dRdb @ normals0

                dDj_da = Dj * (1 - Dj) * 2 * down_k * -dnj_da[-1]
                dDj_db = Dj * (1 - Dj) * 2 * down_k * -dnj_db[-1]

                dUi_da = Ui * (1 - Ui) * 2 * up_k * dnda[-1]
                dUi_db = Ui * (1 - Ui) * 2 * up_k * dndb[-1]

                dl_da = dRda @ rotate2initial(l, R)
                dl_db = dRdb @ rotate2initial(l, R)

                dmask_da[idx] += (np.dot(cast_dir, dl_da) / n_rays * Dj * Ui + np.dot(cast_dir,
                                                                                      l) / n_rays * dDj_da * Ui + np.dot(
                    cast_dir, l) / n_rays * Dj * dUi_da) / v
                dmask_db[idx] += (np.dot(cast_dir, dl_db) / n_rays * Dj * Ui + np.dot(cast_dir,
                                                                                      l) / n_rays * dDj_db * Ui + np.dot(
                    cast_dir, l) / n_rays * Dj * dUi_db) / v

    return mask, dmask_da, dmask_db


def smooth_overhang_connectivity(mesh, rotated_mesh: pv.PolyData | pv.DataSet, R, dRda, dRdb, par: dict) -> tuple:
    connectivity = par['connectivity']
    build_dir = par['build_dir']
    down_thresh = par['down_thresh']
    up_thresh = par['up_thresh']
    down_k = par['down_k']
    up_k = par['up_k']

    # construct upward, downward and combined fields
    Down = smooth_heaviside(-1 * rotated_mesh['Normals'][:, 2], down_k, down_thresh)
    Up = smooth_heaviside(rotated_mesh['Normals'][:, 2], up_k, up_thresh)
    mask = np.zeros_like(Down)

    # set up derivative fields
    dmask_da = np.zeros_like(Down)
    dmask_db = np.zeros_like(Down)

    # loop over cells and subtract value to compensate for overhangs
    for idx in range(rotated_mesh.n_cells):
        # extract points and normal vector from cell
        cell = rotated_mesh.extract_cells(idx)

        normals0 = np.transpose(mesh['Normals'][idx])
        dnda = dRda @ normals0
        dndb = dRdb @ normals0

        Di = Down[idx]
        dDi_da = Di * (1 - Di) * 2 * down_k * -dnda[-1]
        dDi_db = Di * (1 - Di) * 2 * down_k * -dndb[-1]

        Ui = Up[idx]
        dUi_da = Ui * (1 - Ui) * 2 * up_k * dnda[-1]
        dUi_db = Ui * (1 - Ui) * 2 * up_k * dndb[-1]

        mask[idx] += Di
        dmask_da[idx] += dDi_da
        dmask_db[idx] += dDi_db

        # loop over connected cells and add support on part contribution
        center = cell['Center'][0]
        conn = connectivity[idx]
        v = len(conn)

        if v > 0:
            c = rotated_mesh.extract_cells(conn)['Center']
            l = np.subtract(center, c)
            l = l / np.linalg.norm(l, axis=1)[:, None]

            Dj = Down[conn]

            mask_val = np.sum(np.dot(-build_dir, l.transpose()) * Dj) * Ui / v
            mask[idx] += mask_val

            normals0_ = np.transpose(mesh['Normals'][conn])
            dnj_da = np.transpose(dRda @ normals0_)
            dnj_db = np.transpose(dRdb @ normals0_)

            dDj_da = Dj * (1 - Dj) * 2 * down_k * -dnj_da[:, -1]
            dDj_db = Dj * (1 - Dj) * 2 * down_k * -dnj_db[:, -1]

            dl_da = np.transpose(dRda @ rotate2initial(l.transpose(), R))
            dl_db = np.transpose(dRdb @ rotate2initial(l.transpose(), R))

            dm_da_val = np.sum(np.dot(-build_dir, dl_da.transpose()) * Dj * Ui + np.dot(-build_dir, l.transpose()) * dDj_da * Ui + np.dot(-build_dir, l.transpose()) * Dj * dUi_da) / v
            dm_db_val = np.sum(np.dot(-build_dir, dl_db.transpose()) * Dj * Ui + np.dot(-build_dir, l.transpose()) * dDj_db * Ui + np.dot(-build_dir, l.transpose()) * Dj * dUi_db) / v

            dmask_da[idx] += dm_da_val
            dmask_db[idx] += dm_db_val

    return mask, dmask_da, dmask_db


def generate_connectivity(mesh):
    res = []

    # add centroid coordinate to cells
    mesh.cell_data['Center'] = [np.sum(c.points, axis=0) / 3 for c in mesh.cell]

    for i in range(mesh.n_cells):
        arr = []

        for j in range(mesh.n_cells):
            if i == j:
                continue

            # # extract cells and normal vectors
            # ci = mesh.extract_cells(i)
            # cj = mesh.extract_cells(j)

            # check if normals point towards each other
            line = mesh['Center'][j] - mesh['Center'][i]
            line = line/np.linalg.norm(line)
            if np.dot(line, mesh['Normals'][i]) > 0:

                # do ray tracing for check
                _, ids = mesh.ray_trace(mesh['Center'][i], mesh['Center'][i] + 2 * line)

                # check that first intersected cell is j, otherwise append first intersected idx
                ids = ids[ids != i]
                arr.append(ids[0])
        res.append(list(set(arr)))

    return res


def generate_connectivity_obb(mesh):
    res = []

    # add centroid coordinate to cells
    mesh.cell_data['Center'] = [np.sum(c.points, axis=0) / 3 for c in mesh.cell]

    obb = vtkOBBTree()
    obb.SetDataSet(mesh)
    obb.BuildLocator()

    con_elm = vtkCellArray()

    for i in range(mesh.n_cells):
        arr = []

        for j in range(mesh.n_cells):
            if i != j:

                # check if normals point towards each other
                line = mesh['Center'][j] - mesh['Center'][i]
                line = line / np.linalg.norm(line)
                if np.dot(line, mesh['Normals'][i]) > 0:
                    # do ray tracing for check
                    # _, ids = mesh.ray_trace(mesh['Center'][i], mesh['Center'][i] + 2 * line)
                    sec = vtkPoints()

                    code = obb.IntersectWithLine(mesh['Center'][i] + 1e-3*line, mesh['Center'][j] - 1e-3 * line, sec, None)

                    # check that first intersected cell is j, otherwise append first intersected idx
                    # ids = ids[ids != i]
                    if sec.GetData().GetNumberOfTuples() == 0:
                        arr.append(j)
        res.append(list(set(arr)))

    return res
