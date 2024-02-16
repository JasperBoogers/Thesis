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

top_cover, lines = extract_top_cover(upward)

# plot
p = pv.Plotter()
_ = p.add_mesh(upward, show_edges=True, color='g', opacity=0.5)
_ = p.add_mesh(top_cover, show_edges=True, color='r', opacity=0.5)
for line in lines:
    _ = p.add_mesh(line, color='b')
p.show()