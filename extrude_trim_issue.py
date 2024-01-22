import pyvista as pv
import numpy as np
from pyvista import examples

# download mesh and compute normals
mesh = examples.download_cow()
mesh.triangulate(inplace=True).compute_normals(inplace=True)

# extract overhanging surfaces
ids = np.arange(mesh.n_cells)[mesh['Normals'][:, 2] < -0.7]
overhang = mesh.extract_cells(ids).extract_surface()
overhang.triangulate(inplace=True)

# create plane to extrude to
bounds = overhang.bounds
plane = pv.Plane(center=(0, 0, bounds[-2] - 1),
                 i_size=1.1 * (bounds[1] - bounds[0]),
                 j_size=1.1 * (bounds[3] - bounds[2]),
                 direction=(0, 0, 1))

# extrude the overhanging surface to the plane
support = overhang.extrude_trim((0, 0, -1), plane)
support.triangulate(inplace=True).compute_normals(inplace=True)

# extract boundary edges
edges = support.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)

# check
print(f'number of open edges: {support.n_open_edges}')  # 212
print(f'is the support manifold: {support.is_manifold}')  # False

# plot
plot = pv.Plotter()
_ = plot.add_mesh(mesh, opacity=0.5, color='blue')
_ = plot.add_mesh(overhang, opacity=1, color='orange')
_ = plot.add_mesh(support, opacity=0.5, color='green')
_ = plot.add_mesh(edges, color='r', line_width=3)
plot.show()

# attempt to repair the mesh
import pymeshfix as mf
meshfix = mf.MeshFix(support)
meshfix.repair(joincomp=True, remove_smallest_components=False)
support_repaired = meshfix.mesh.triangulate()

# check
print(f'number of open edges: {support_repaired.n_open_edges}')  # 0
print(f'is the support manifold: {support_repaired.is_manifold}')  # True

# plot again
plot = pv.Plotter()
_ = plot.add_mesh(support, opacity=0.5, color='green')
_ = plot.add_mesh(support_repaired, opacity=1, color='purple')
plot.show()
print(overhang)
