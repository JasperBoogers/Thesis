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


def extract_top_cover(m, x_res: float = None, y_res: float = None):
    # compute average cell area and sampling cell length
    A_avg = m.area / m.n_cells
    bounds = m.bounds

    L = np.sqrt(A_avg) / 4
    if x_res is not None:
        x_coords = np.linspace(bounds[0] + x_res / 2, bounds[1] - x_res / 2, int((bounds[1] - bounds[0]) / x_res))
    else:
        x_coords = np.linspace(bounds[0] + L / 2, bounds[1] - L / 2, int((bounds[1] - bounds[0]) / L))

    if y_res is not None:
        y_coords = np.linspace(bounds[2] + y_res / 2, bounds[3] - y_res / 2, int((bounds[3] - bounds[2]) / y_res))
    else:
        y_coords = np.linspace(bounds[2] + L / 2, bounds[3] - L / 2, int((bounds[3] - bounds[2]) / L))

    z_min = 1.1 * bounds[-2]
    z_max = 1.1 * bounds[-1]

    top_idx = set()
    lines = []

    for x in x_coords:
        for y in y_coords:

            # make a line
            line = pv.Line([x, y, z_min], [x, y, z_max])
            lines.append(line)

            # check if any cells intersect that line
            intersect = m.find_cells_intersecting_line(line.points[0], line.points[1])

            if len(intersect) > 1:

                # calculate average center coordinate of each intersecting cell
                points = np.array([m.extract_cells(i).points for i in intersect])
                centers = np.sum(points, axis=1)

                # add cell idx with highest z-coordinate to top_idx
                max_idx = np.argmax(centers[:, -1])
                top_idx.add(intersect[max_idx])
            elif len(intersect) > 0:
                # only one intersecting cell -> top cover
                top_idx.add(intersect[0])
            else:
                pass

    return m.extract_cells(list(top_idx)), lines
