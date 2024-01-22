# from scipy.spatial.transform import Rotation
import numpy as np
from scipy.optimize import minimize
import time
import jax.numpy as jnp
from jax import value_and_grad, jit
import matplotlib.pyplot as plt


def support_2D_Euler(t, points, faces, normals, proj):
    t = t[0]

    # set up (derivative of) rotation matrix
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    dR = np.array([[-np.sin(t), -np.cos(t)], [np.cos(t), -np.sin(t)]])

    # rotate points
    points_rot = R @ points
    dpoints = dR @ points
    normals_rot = R @ np.transpose(normals)

    # determine the lowest point to project to
    z_min = np.min(points_rot) - proj

    # identify downward facing normals
    overhang_idx = np.arange(faces.shape[0])[normals_rot[1, :] < -1e-6]

    S = 0
    dS = 0

    for idx in overhang_idx:
        # extract points corresponding to idx
        p1 = points_rot[:, faces[idx][0]]
        dp1 = dpoints[:, faces[idx][0]]
        p2 = points_rot[:, faces[idx][1]]
        dp2 = dpoints[:, faces[idx][1]]

        # calculate A & h, multiply to get S
        A = p2[0] - p1[0]
        h = (p2[-1] - p1[-1])/2 - z_min
        S += A*h

        # calculate dA & dh to get dS
        dA = dp2[0] - dp1[0]
        dh = (dp2[-1] - dp1[-1])/2
        dS += (dA * h + A * dh)

    return -S, -dS


if __name__ == "__main__":

    # set mesh and projection plane
    p = np.array([[-1, 1, 1, -1], [-1, -1, 1, 1]])
    f = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    n = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])
    plane = 0

    angles = np.linspace(-2*np.pi, 2*np.pi, 201)
    support = np.zeros_like(angles)
    dSdt = np.zeros_like(angles)
    for k, angle in enumerate(angles):
        s, ds = support_2D_Euler([angle], p, f, n, plane)
        support[k] = s
        dSdt[k] = ds

    fig = plt.figure()
    plt.plot(angles, support, 'r')
    plt.plot(angles, dSdt, 'b')
    plt.plot(angles, 2*dSdt, 'g')
    plt.plot(angles, -np.sin(4*angles)/np.abs(np.cos(angles)*np.sin(angles)), 'r.')
    plt.legend(['Support volume', 'dSdt', 'Analytic'])
    plt.ylim([-4, 4])
    plt.xlim([-2*np.pi, 2*np.pi])
    plt.show()

    # initial conditions
    t0 = np.array([np.pi/10])

    print('Solve using Euler angles')
    start = time.time()
    res_euler = minimize(support_2D_Euler, t0, args=(p, f, n, plane),
                         jac=True, options={'disp': True})
    t_euler = time.time() - start
    print(res_euler)
    print(f'Execution time: {t_euler}')
