import numpy as np
import pyvista as pv
from time import time, sleep
# from os import path
from scipy.spatial.transform import Rotation
from vtk import vtkTrimmedExtrusionFilter


# set parameters
ORIGIN = np.array([0, 0, 0])
OVERHANG_THRESHOLD = np.deg2rad(0)
PROJ = -2
ROTATION = [np.pi/4, 0, 0]

# create a pv Plotter and show axis system
plot = pv.Plotter()  # type: ignore
plot.add_axes(shaft_length=0.9)
plot.add_axes_at_origin()

# create mesh
mesh = pv.read('Geometries/cube.stl')
assert mesh is not None
mesh.compute_normals(inplace=True)

# rotate mesh and add to plot
tfm = np.identity(4)
rot = Rotation.from_euler('xyz', ROTATION)
tfm[:-1, :-1] = rot.as_matrix()
mesh.transform(tfm)
plot.add_mesh(mesh, style='wireframe', color='green')
plot.show(interactive_update=True)

# calc support volume, plane size should exceed model bounds! #TODO
plane = pv.Plane(center=(0, 0, PROJ), i_size=10, j_size=10, direction=(0, 0, 1))

overhang_idx = np.arange(mesh.n_cells)[mesh['Normals'][:, 2] < 0.0]  # type: ignore
overhang = mesh.extract_cells(overhang_idx)
assert overhang is not None
overhang = overhang.extract_surface()
assert overhang is not None

SV = overhang.extrude_trim((0, 0, -1), plane)
assert SV is not None
SV.triangulate(inplace=True)
SV.compute_normals(inplace=True, flip_normals=True)
plot.add_mesh(SV, opacity=0.5, color='red', show_edges=True)
print(f'volume under object: {SV.volume}')
plot.show()
