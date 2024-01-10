# from scipy.spatial.transform import Rotation
import numpy as np
from scipy.optimize import minimize
import time
import jax.numpy as jnp
from jax import value_and_grad, jit, grad


def support_2D_Euler(t, pts, proj):
    t = t[0]
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    points_rot = R @ pts

    # extract points
    p1, p2, p3, p4 = np.split(points_rot, 4, axis=1)

    # calculate support volume
    S1 = 0.5*(p1[0] - p4[0]) * (p4[1] - p1[1])
    S2 = (p1[0] - p4[0]) * (p1[1] - proj)
    S3 = (p2[0] - p1[0]) * (p1[1] - proj)
    S4 = 0.5*(p2[0] - p1[0]) * (p2[1] - p1[1])
    S = S1 + S2 + S3 + S4

    # calculate derivative
    dSdt_analytic = 2*proj*np.sin(t) - 2*proj*np.cos(t)
    Pz = np.array([[0, 1], [-1, 0]])
    dM = - Pz @ R
    PR = dM @ pts
    # extract points
    dp1, dp2, dp3, dp4 = np.split(PR, 4, axis=1)

    # calculate support volume
    dS1 = 0.5 * (dp1[0] - dp4[0]) * (dp4[1] - dp1[1])
    dS2 = (dp1[0] - dp4[0]) * (dp1[1] - proj)
    dS3 = (dp2[0] - dp1[0]) * (dp1[1] - proj)
    dS4 = 0.5 * (dp2[0] - dp1[0]) * (dp2[1] - dp1[1])
    dSdt = dS1 + dS2 + dS3 + dS4

    # set correct dtypes for output
    S = S[0]
    dSdt = np.array(dSdt_analytic)
    return -S, -dSdt


@jit
def support_2D_jax(t, pts, proj):
    t = t[0]
    R = jnp.array([[jnp.cos(t), -jnp.sin(t)], [jnp.sin(t), jnp.cos(t)]])
    points_rot = R @ pts

    # extract points
    p1, p2, p3, p4 = jnp.split(points_rot, 4, axis=1)

    # calculate support volume
    S1 = 0.5 * (p1[0] - p4[0]) * (p4[1] - p1[1])
    S2 = (p1[0] - p4[0]) * (p1[1] - proj)
    S3 = (p2[0] - p1[0]) * (p1[1] - proj)
    S4 = 0.5 * (p2[0] - p1[0]) * (p2[1] - p1[1])
    S = S1 + S2 + S3 + S4

    # set correct dtypes for output
    S = S[0]

    return -S


def support_2D_jax_grad(t, pts, plane):
    return grad(support_2D_jax, 0)(t, pts, plane)


if __name__ == "__main__":

    # set mesh and projection plane
    points = np.array([[-1, 1, 1, -1], [-1, -1, 1, 1]])
    plane = -1

    # initial conditions
    t0 = [np.pi/10]

    # print('Solve using Euler angles')
    start = time.time()
    # res_euler = minimize(support_2D_Euler, t0, args=(points, plane),
    #                      jac=True, options={'disp': True})
    t_euler = time.time() - start
    # print(res_euler)

    print('Solve using JAX')
    start = time.time()
    res_jax = minimize(support_2D_jax, t0, args=(points, plane),
                       jac=support_2D_jax_grad, options={'disp': True})
    t_jax = time.time() - start
    print(res_jax)
    print(f'Analytic: {t_euler} seconds, Jax: {t_jax}')
