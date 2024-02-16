import pyvista as pv
import numpy as np
from helpers import *

# load file and rotate
FILE = 'Geometries/chair.stl'
mesh = pv.read(FILE)
mesh = prep_mesh(mesh.rotate_x(45, inplace=True))

# extract upward facing triangles
thresh = 1e-6
upward_idx = np.arange(mesh.n_cells)[mesh['Normals'][:, 2] > thresh]
upward = mesh.extract_cells(upward_idx)

# compute average cell area and sampling cell length
A_avg = upward.area/upward.n_cells
L = np.sqrt(A_avg)/4
bounds = upward.bounds
x_coords = np.linspace(bounds[0] + L/2, bounds[1] - L/2, int((bounds[1] - bounds[0])/L))
y_coords = np.linspace(bounds[2] + L/2, bounds[3] - L/2, int((bounds[3] - bounds[2])/L))

z_min = 1.1*bounds[-2]
z_max = 1.1*bounds[-1]

top_idx = set()
lines = []

for x in x_coords:
    for y in y_coords:

        # make a line
        line = pv.Line([x, y, z_min], [x, y, z_max])
        lines.append(line)

        # check if any cells intersect that line
        intersect = upward.find_cells_intersecting_line(line.points[0], line.points[1])

        if len(intersect) > 1:

            # calculate average center coordinate of each intersecting cell
            points = np.array([upward.extract_cells(i).points for i in intersect])
            centers = np.sum(points, axis=1)

            # add cell idx with highest z-coordinate to top_idx
            max_idx = np.argmax(centers[:, -1])
            top_idx.add(intersect[max_idx])
        elif len(intersect) > 0:
            # only one intersecting cell -> top cover
            top_idx.add(intersect[0])
        else:
            pass

top_cover = upward.extract_cells(list(top_idx))

# plot
p = pv.Plotter()
_ = p.add_mesh(upward, show_edges=True, color='g', opacity=0.5)
_ = p.add_mesh(top_cover, show_edges=True, color='r', opacity=0.5)
for line in lines:
    _ = p.add_mesh(line, color='b')
p.show()