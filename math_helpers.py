import numpy as np


def finite_forward_differences(y, x) -> np.ndarray:
    h = (x[-1] - x[0])/len(x)
    return np.diff(y)/h


def construct_skew_matrix(x: float | int, y: float | int, z: float | int) -> np.ndarray:
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def cross_product(v1: np.ndarray | list, v2: np.ndarray | list) -> np.ndarray:
    return np.cross(v1, v2)


def rotate2initial(v, mat) -> np.ndarray:
    return np.transpose(mat) @ v


def construct_rotation_matrix(a, b) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    R = Ry @ Rx

    # construct derivatives of rotation matrices
    dRx = construct_skew_matrix(1, 0, 0) @ Rx
    dRy = construct_skew_matrix(0, 1, 0) @ Ry
    dRdx = Ry @ dRx
    dRdy = dRy @ Rx

    return Rx, Ry, R, dRdx, dRdy


def smooth_heaviside(x: float, k: float, x0: float) -> float:
    H = 1/(1 + np.exp(-2 * k * (x - x0)))
    return H
