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

    # move mesh center of bounding box to origin
    center = np.array(m.center)
    m.translate(-center, inplace=True)
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


def make_contour_plot(x: np.ndarray, y: np.ndarray, f: np.ndarray, save=None):
    x, y = np.meshgrid(x, y)
    fig = plt.figure()
    cp = plt.contourf(x, y, f)
    fig.colorbar(cp)
    plt.xlabel('Rotation about x-axis [deg]')
    plt.ylabel('Rotation about y-axis [deg]')
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
