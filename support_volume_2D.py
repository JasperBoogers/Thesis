# from scipy.spatial.transform import Rotation
import numpy as np
from scipy.optimize import minimize


def support_volume_2D(t, pts, proj):
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

    # set correct types for output
    S = S[0]
    dSdt = np.array(dSdt)
    return -S, -dSdt


points = np.array([[-1, 1, 1, -1], [-1, -1, 1, 1]])
plane = -4  # projection plane y-coordinate
t0 = [np.pi/10]

res = minimize(support_volume_2D, t0, args=(points, plane),
               jac=True, options={'disp': True})
print(res)
