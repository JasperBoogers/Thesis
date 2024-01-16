# from scipy.spatial.transform import Rotation
import numpy as np
from scipy.optimize import minimize
import time
import jax.numpy as jnp
from jax import value_and_grad, jit
import matplotlib.pyplot as plt


def support_2D_Euler(t, pts, fcs, n, proj):
    t = t[0]
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    points_rot = R @ pts
    normals_rot = R @ np.transpose(n)

    # calculate face lengths
    lengths = [np.linalg.norm(pts[:, j] - pts[:, i]) for (i, j) in fcs]

    # determine the lowest point to project to
    z_min = np.min(points_rot) - proj

    # identify downward facing normals
    overhang_idx = np.arange(fcs.shape[0])[normals_rot[1, :] < -1e-6]
    downward_faces = fcs[overhang_idx]

    S = 0
    for idx in overhang_idx:
        p1, p2 = points_rot[:, faces[idx][0]], points_rot[:, faces[idx][1]]
        A = p2[0] - p1[0]
        # A = abs(lengths[idx] * np.cos(t))
        h = (p2[-1] + p1[-1])/2 - z_min
        S += A*h

    return -S


@jit
def support_2D_jax(t, pts, proj):
    t = t[0]
    R = jnp.array([[jnp.cos(t), -jnp.sin(t)], [jnp.sin(t), jnp.cos(t)]])
    points_rot = R @ pts

    # extract points
    p1, p2, p3, p4 = jnp.split(points_rot, 4, axis=1)

    # calculate support volume
    S1 = 0.5*(p1[0] - p4[0]) * (p4[1] - p1[1])
    S2 = (p1[0] - p4[0]) * (p1[1] - proj)
    S3 = (p2[0] - p1[0]) * (p1[1] - proj)
    S4 = 0.5*(p2[0] - p1[0]) * (p2[1] - p1[1])
    S = S1 + S2 + S3 + S4

    # calculate derivative
    # dSdt = 2*proj*np.sin(t) - 2*proj*np.cos(t)

    # set correct dtypes for output
    S = S[0]

    return -S


def support_2D_jax_grad(t, pts, plane):
    return value_and_grad(support_2D_jax, 0)(t, pts, plane)


if __name__ == "__main__":

    # set mesh and projection plane
    points = np.array([[-1, 1, 1, -1], [-1, -1, 1, 1]])
    faces = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    normals = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])
    plane = 1

    angles = np.linspace(-np.pi, np.pi, 101)
    S = np.zeros_like(angles)
    for n, angle in enumerate(angles):
        S[n] = support_2D_Euler([angle], points, faces, normals, plane)

    fig = plt.figure()
    plt.plot(np.rad2deg(angles), S)
    plt.show()

    # initial conditions
    t0 = [np.pi/10]

    print('Solve using Euler angles')
    start = time.time()
    res_euler = minimize(support_2D_Euler, t0, args=(points, faces, normals, plane),
                         jac='3-point', options={'disp': True})
    t_euler = time.time() - start
    print(res_euler)
    print(f'Execution time: {t_euler}')
