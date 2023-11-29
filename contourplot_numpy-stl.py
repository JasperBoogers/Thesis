from matplotlib import pyplot as plt
import numpy as np
from stl import mesh
from mpl_toolkits import mplot3d
import time
import os
from f_eval import f_eval


# load mesh
file = '3DBenchy-bridge.stl'
fp = os.path.join('Geometries', '3DBenchy', file)
m = mesh.Mesh.from_file(fp)
print(f'Loaded file {file} with {m.vectors.shape[0]} faces')

# set up iteration variables
angle = np.pi
num = 20
theta = phi = np.linspace(-angle, angle, num)
f = np.zeros((num, num))

start = time.time()
for i, t in enumerate(theta):
    for j, p in enumerate(phi):
        _, f[i, j] = f_eval(m, t, p)
end = time.time()
print(f'execution duration: {end-start} seconds')

opt_idx = np.unravel_index(np.argmin(f), f.shape)
m_rot = mesh.Mesh(m.data.copy())
m_rot.rotate([1, 0, 0], theta[opt_idx[0]])
m_rot.rotate([0, 1, 0], phi[opt_idx[1]])

fig, ax = plt.subplots()
x, y = np.meshgrid(theta, phi)
cs = ax.contour(x, y, f)
ax.clabel(cs, inline=True, fontsize=10)
ax.set_xlabel('x axis rotation')
ax.set_ylabel('y axis rotation')
ax.set_title(f'Contour plot of CoG z-value of {file}')

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.add_collection3d(mplot3d.art3d.Poly3DCollection(m.vectors))
ax2.add_collection3d(mplot3d.art3d.Poly3DCollection(m_rot.vectors, facecolors='g'))
scale = m.points.flatten()
ax2.auto_scale_xyz(scale, scale, scale)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title(f'Optimal orientation of {file} \n t={round(theta[opt_idx[0]], 1)}, p={round(phi[opt_idx[1]])}')
plt.show()
