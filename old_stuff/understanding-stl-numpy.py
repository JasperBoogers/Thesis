from matplotlib import pyplot as plt
from stl import mesh
from mpl_toolkits import mplot3d
import math
import numpy as np

# Create 3 faces of a cube, 2 triangles per face -> triangles
data = np.zeros(6, dtype=mesh.Mesh.dtype)

# Top of the cube consists of two triangles, so needs two items in 'vectors'
data['vectors'][0] = np.array([[0, 1, 1],
                                [1, 0, 1],
                                [0, 0, 1]])
data['vectors'][1] = np.array([[1, 0, 1],
                                  [0, 1, 1],
                                  [1, 1, 1]])

cube = mesh.Mesh(data)

# plotting
fig = plt.figure()
axes = fig.add_subplot(projection='3d')
axes.add_collection3d(mplot3d.art3d.Poly3DCollection(cube.vectors))
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_zlabel('z')
plt.show()
