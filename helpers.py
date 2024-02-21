import pyvista as pv
import numpy as np
from scipy.spatial.transform import Rotation
from matplotlib import pyplot as plt


def prep_mesh(m: pv.PolyData | pv.DataSet, decimation=0.9, flip=False) -> pv.PolyData:
    # ensure mesh is only triangles
    m.triangulate(inplace=True)

    # decimate mesh by decimate*100%
    m.decimate_pro(decimation)

    # (re)compute normals, and flip normal direction if needed
    m.compute_normals(inplace=True, flip_normals=flip)

    # move mesh center of mass to origin
    m.translate(-m.center_of_mass(), inplace=True)
    return m


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
    return np.linalg.norm([x, y, z])/2


def finite_forward_differences(y, x):
    h = (x[-1] - x[0])/len(x)
    return np.diff(y)/h


def construct_skew_matrix(x: float | int, y: float | int, z: float | int) -> np.ndarray:
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def cross_product(v1: np.ndarray | list, v2: np.ndarray | list) -> np.ndarray:
    return np.cross(v1, v2)


def extract_top_cover(m):

    # set bounds
    bounds = m.bounds
    z_min = bounds[-2] - 5
    z_max = bounds[-1] + 5

    # pre-calculate cell centers
    # compute average coordinate for each cell, and store in 'Center' array
    m.cell_data['Center'] = [np.sum(c.points, axis=0) / 3 for c in m.cell]

    top = set()
    not_top = set()
    lines = []

    for i in range(m.n_cells):

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
            centers = m.extract_cells(intersect)['Center']
            max_idx = np.argmax(centers[:, -1])

            # add cell index with max z to top
            top.add(intersect[max_idx])

            # add other cells to not_top
            not_top.update(np.delete(intersect, max_idx))
        elif len(intersect) > 0:
            # only one intersecting cell -> top cover
            top.update(intersect)
        else:
            pass

    return m.extract_cells(list(top)), lines


def construct_rotation_matrix(ax, ay):
    Rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    Ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    R = Ry @ Rx

    # construct derivatives of rotation matrices
    dRx = construct_skew_matrix(1, 0, 0) @ Rx
    dRy = construct_skew_matrix(0, 1, 0) @ Ry
    dRdx = Ry @ dRx
    dRdy = dRy @ Rx

    return Rx, Ry, R, dRdx, dRdy


def rotate2initial(v, mat):
    return np.transpose(mat) @ v


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
