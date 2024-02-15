import pyvista as pv
import numpy as np
from helpers import *

# load file and rotate
FILE = 'Geometries/chair.stl'
mesh = pv.read(FILE)
mesh = prep_mesh(mesh.rotate_x(90, inplace=True))

# extract upward facing triangles
thresh = 1e-6
upward_idx = np.arange(mesh.n_cells)[mesh['Normals'][:, 2] > thresh]
upward = mesh.extract_cells(upward_idx)

# compute average cell area and sampling cell length
A_avg = upward.area/upward.n_cells
L = np.sqrt(A_avg)/2
bounds = upward.bounds
x_coords = np.linspace(bounds[0] + L/2, bounds[1] - L/2, int((bounds[1] - bounds[0])/L))
y_coords = np.linspace(bounds[2] + L/2, bounds[3] - L/2, int((bounds[3] - bounds[2])/L))

z_min = 1.1*bounds[-2]
z_max = 1.1*bounds[-1]

lines = []
for x in x_coords:
    for y in y_coords:
        line = pv.Line([x, y, z_min], [x, y, z_max])
        lines.append(line)

# plot
p = pv.Plotter()
_ = p.add_mesh(upward, color='g', opacity=0.5)
for line in lines:
    _ = p.add_mesh(line, color='b')
p.show()