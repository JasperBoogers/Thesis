# from scipy.spatial.transform import Rotation
import numpy as np
from scipy.optimize import minimize
import time
import jax.numpy as jnp
from jax import value_and_grad, jit


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
    dSdt = 2*proj*np.sin(t) - 2*proj*np.cos(t)

    # set correct dtypes for output
    S = S[0]
    dSdt = np.array(dSdt)
    return -S, -dSdt


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
    plane = -4

    # initial conditions
    t0 = [np.pi/10]

    print('Solve using Euler angles')
    start = time.time()
    res_euler = minimize(support_2D_Euler, t0, args=(points, plane),
                         jac=True, options={'disp': True})
    t_euler = time.time() - start
    print(res_euler)

    print('Solve using JAX')
    start = time.time()
    res_jax = minimize(support_2D_jax_grad, t0, args=(points, plane),
                       jac=True, options={'disp': True})
    t_jax = time.time() - start
    print(res_jax)
    print(f'Analytic: {t_euler} seconds, Jax: {t_jax}')
