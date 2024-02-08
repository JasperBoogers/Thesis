# from scipy.spatial.transform import Rotation
import numpy as np
from scipy.optimize import minimize
import time
import jax.numpy as jnp
from jax import value_and_grad, jit
from matplotlib import pyplot as plt
import matplotlib as mpl
from latex_params import latex_params
# mpl.rcParams.update(latex_params['params'])
mpl.rcParams['text.usetex'] = False


def finite_differences(y, x):
    h = (x[-1] - x[0])/len(x)
    return np.diff(y)/h


def support_2D(t, points, faces, normals, proj):
    t = t[0]

    # set up (derivative of) rotation matrix
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    dR = np.array([[-np.sin(t), -np.cos(t)], [np.cos(t), -np.sin(t)]])

    # rotate points
    points_rot = R @ points
    dpoints = dR @ points
    normals_rot = R @ np.transpose(normals)

    # determine the lowest point to project to
    if proj > 0:  # fixed projection height
        z_min = np.array([0, -proj])
        dz_min = dR @ (np.transpose(R) @ z_min)
    elif proj < 0:  # projection height is the lowest point
        z_min = points_rot[:, np.argmin(points_rot[-1, :])]
        dz_min = dR @ (np.transpose(R) @ z_min)
    else:  # no projecting down
        z_min = [0]
        dz_min = [0]

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

        # calculate A & dA
        A = p2[0] - p1[0]
        dA = dp2[0] - dp1[0]

        if proj == 0:
            # calculate h & dh, order of points matters!
            if p2[-1] < p1[-1]:
                h = (p1[-1] - p2[-1])/2
                dh = (dp1[-1] - dp2[-1])/2
            else:
                h = (p2[-1] - p1[-1])/2
                dh = (dp2[-1] - dp1[-1]) / 2
        else:
            h = (p2[1]+p1[1])/2 - z_min[-1]
            dh = (dp2[-1] + dp1[-1]) / 2 - dz_min[-1]

        # multiply A, h, dA & dh to get S & dS
        S += A * h
        dS += (dA * h + A * dh)

    return S, dS, z_min[-1]


if __name__ == "__main__":

    # set mesh and projection plane
    p = np.array([[-1/2, 1/2, 1/2, -1/2], [-1/2, -1/2, 1/2, 1/2]])
    f = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    n = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])
    plane = -1

    step = 201
    angles = np.linspace(-1*np.pi, 1*np.pi, step)
    support = np.zeros_like(angles)
    dSdt = np.zeros_like(angles)
    lowest_z = np.zeros_like(angles)
    for k, angle in enumerate(angles):
        s, ds, z = support_2D([angle], p, f, n, plane)
        support[k] = s
        dSdt[k] = ds
        lowest_z[k] = z

    fig = plt.figure()
    plt.plot(angles, support, 'g', label='Support')
    plt.plot(angles, dSdt, 'b.', label=r"Calculated derivative")
    plt.plot(angles[:-1], finite_differences(support, angles), 'r.', label='Finite difference')
    plt.plot(angles, lowest_z, 'r', label='Lowest y-coordinate')
    # plt.plot(angles, np.sin(4*angles)/np.abs(np.cos(angles)*np.sin(angles)), 'r')
    plt.xlabel('Angle [rad]')
    plt.ylabel('Magnitude [-]')
    plt.title('Rotating a square - no projection')
    plt.legend(loc='upper right')
    # plt.savefig('out/supportvolume/2D_solution_no_proj.svg', format='svg', bbox_inches='tight')
    plt.show()

    # initial conditions
    t0 = np.array([np.pi/10])

    start = time.time()
    res_euler = minimize(support_2D, t0, args=(p, f, n, plane), method='BFGS',
                         jac=True, options={'disp': True})
    t_euler = time.time() - start
    print(res_euler)
    print(f'Execution time: {t_euler}')
