import numpy as np
import pyvista as pv
from time import time, sleep
# from os import path
from scipy.spatial.transform import Rotation
from vtk import vtkTrimmedExtrusionFilter


def calc_overhang(v):
    return np.dot(v, [0, 0, 1])


# set parameters
ORIGIN = np.array([0, 0, 0])
OVERHANG_THRESHOLD = np.deg2rad(0)
PROJ = -2
ROTATION = [np.pi/4, 0, 0]

# create a pv Plotter and show axis system
plot = pv.Plotter()  # type: ignore
plot.add_axes(shaft_length=0.9)

# define cube points in undeformed config
x_coords = [-1, 1, 1, -1, -1, 1, 1, -1]
y_coords = [-1, -1, 1, 1, -1, -1, 1, 1]
z_coords = [-1, -1, -1, -1, 1, 1, 1, 1]
points = np.array([x_coords, y_coords, z_coords])
faces = np.hstack([[4, 0, 1, 2, 3],
                  [4, 4, 5, 6, 7],
                  [4, 1, 2, 6, 5],
                  [4, 0, 3, 7, 4],
                  [4, 0, 1, 5, 4],
                  [4, 3, 2, 6, 7]])

# create mesh
mesh = pv.PolyData(points.transpose(), faces)
mesh.compute_normals(inplace=True, flip_normals=True)
area = mesh.compute_cell_sizes(length=False, volume=False)

# rotate mesh and add to plot
tfm = np.identity(4)
rot = Rotation.from_euler('xyz', ROTATION)
tfm[:-1, :-1] = rot.as_matrix()
mesh.transform(tfm)
plot.add_mesh(mesh, show_edges=True, opacity=0.5, color='green')

# calculate base plate and add to plot
projection = mesh.project_points_to_plane(origin=[0, 0, PROJ])
plot.add_mesh(projection, color='blue', name='projection')
plot.show(interactive_update=True)

# calculate the amount of overhang
# dot = np.apply_along_axis(calc_overhang, 1, mesh.cell_normals)
# norm = np.apply_along_axis(np.linalg.norm, 1, mesh.cell_normals)
# overhang = np.divide(dot, norm)
# overhang_idx = np.where(overhang < OVERHANG_THRESHOLD)

# # extract cells which are pointing downwards
# overhang_idx = np.arange(mesh.n_cells)[mesh['Normals'][:, 2] < 0.0]
# overhang = mesh.extract_cells(overhang_idx)
# plot.add_mesh(overhang, color='red')



print('finish')
