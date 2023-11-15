from matplotlib import pyplot as plt
from stl import mesh
from mpl_toolkits import mplot3d
import math

'''
see https://pythonhosted.org/numpy-stl/stl.html

integrate with https://plotly.com/python/3d-mesh/ ?

'''


def f_eval(obj, a1, a2):

    # rotate obj sequentially by a1 and a2
    obj_rot = obj.rotate([1, 0, 0], math.radians(a1))
    obj_rot = obj_rot.rotate([0, 1, 1], math.radians(a2))

    # perform function evaluation
    _, cog, _ = obj_rot.get_mass_properties()
    f = cog
    return obj_rot, f


# Create a new plot
figure = plt.figure()
axes = figure.add_subplot(111, projection='3d')

# Load the STL files and add the vectors to the plot
your_mesh = mesh.Mesh.from_file('3DBenchy-stern.stl')
print(f'N of faces: {your_mesh.vectors.shape[0]}')

axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors))

copy = mesh.Mesh(your_mesh.data.copy())
copy.rotate([0, 0, 1], math.radians(90))

# move copy to (0,0,0)
_, CoG, _ = copy.get_mass_properties()
copy.translate(-CoG)

axes.add_collection3d(mplot3d.art3d.Poly3DCollection(copy.vectors, facecolors='g'))

# Auto scale to the mesh size
scale = your_mesh.points.flatten()
axes.auto_scale_xyz(scale, scale, scale)

# Show the plot to the screen
axes.view_init(azim=120)
plt.show()
