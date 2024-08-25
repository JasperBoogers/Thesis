from helpers.helpers import *


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
    if proj is None:  # no projecting down
        z_min = [0]
        dz_min = [0]
    elif proj == 0:  # projection height is the lowest point
        z_min = points_rot[:, np.argmin(points_rot[-1, :])]
        dz_min = dR @ (np.transpose(R) @ z_min)
    else:  # fixed projection height
        z_min = np.array([0, -proj])
        dz_min = dR @ (np.transpose(R) @ z_min)

    # identify downward facing normals
    overhang_idx = np.arange(faces.shape[0])[normals_rot[1, :] < -1e-6]

    S = 0
    dS = 0

    for idx in overhang_idx:  # TODO: vectorize
        # extract points corresponding to idx
        p1 = points_rot[:, faces[idx][0]]
        dp1 = dpoints[:, faces[idx][0]]
        p2 = points_rot[:, faces[idx][1]]
        dp2 = dpoints[:, faces[idx][1]]

        # calculate A & dA
        A = p2[0] - p1[0]
        dA = dp2[0] - dp1[0]

        if proj is None:
            # calculate h & dh, order of points matters!
            if p2[-1] < p1[-1]:
                h = (p1[-1] - p2[-1]) / 2
                dh = (dp1[-1] - dp2[-1]) / 2
            else:
                h = (p2[-1] - p1[-1]) / 2
                dh = (dp2[-1] - dp1[-1]) / 2
        else:
            h = (p2[1] + p1[1]) / 2 - z_min[-1]
            dh = (dp2[-1] + dp1[-1]) / 2 - dz_min[-1]

        # multiply A, h, dA & dh to get S & dS
        S += A * h
        dS += (dA * h + A * dh)

    return S, dS, z_min[-1]


if __name__ == "__main__":

    # set mesh and projection plane
    p = np.array([[-1 / 2, 1 / 2, 1 / 2, -1 / 2], [-1 / 2, -1 / 2, 1 / 2, 1 / 2]])
    f = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    n = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])

    step = 201
    angles = np.linspace(-1 * np.pi, 1 * np.pi, step)

    support = np.zeros_like(angles)
    support_adap = np.zeros_like(angles)

    dSdt = np.zeros_like(angles)
    dSdt_adap = np.zeros_like(angles)
    lowest_z = np.zeros_like(angles)
    for k, angle in enumerate(angles):
        s, ds, z = support_2D([angle], p, f, n, None)
        s_a, ds_a, _ = support_2D([angle], p, f, n, 0)

        support[k] = s
        support_adap[k] = s_a

        dSdt[k] = ds
        dSdt_adap[k] = ds_a

        lowest_z[k] = z

    # substitute nan values
    nan_idx = np.where(np.abs(np.diff(dSdt)) >= 0.5)[0]
    nan_idx = np.append(nan_idx, [nan_idx - 1, nan_idx + 1, nan_idx + 2])
    pos = np.append(nan_idx, [1, -2, -1])
    x = np.insert(angles, pos, np.nan)
    dSdt = np.insert(dSdt, pos, np.nan)

    # plotting fixed projection
    fig, ax = plt.subplots(1, 1)
    ax.plot(np.rad2deg(angles), support, 'b', label='General solution')
    ax.plot(np.rad2deg(angles), np.abs(np.sin(angles) * np.cos(angles)), 'r.', label='Specific solution')
    # ax.plot(np.rad2deg(angles), finite_central_differences(support, angles), 'r.', label='Finite differences')
    ax.set_xlabel(r'$\theta$ [deg]')
    ax.set_ylabel(r'A [mm$^2$]')
    ax.legend()
    # fig.suptitle('Rotating a square, projection to y=-1')
    plt.savefig('out/supportvolume/2D/2D_solution_comp.svg', format='svg')
    plt.show()

    _, ax = plt.subplots(1, 1)
    ax.plot(np.rad2deg(x), dSdt, 'b', label='General solution')
    ax.plot(np.rad2deg(angles), np.sin(4 * angles) / 2 / np.abs(np.sin(2 * angles)), 'r.', label='Specific solution')
    ax.plot(np.rad2deg(angles), finite_central_differences(support, angles), 'r.', label='Finite differences')
    ax.set_xlabel(r'$\theta$ [deg]')
    ax.set_ylabel(r'A$_{,\theta}$ [mm$^2$/deg]')
    ax.legend()
    # fig.suptitle('Rotating a square, projection to y=-1')
    plt.savefig('out/supportvolume/2D/2D_derivative_comp.svg', format='svg')
    plt.show()

    # comparison between general and specific solution, no projection #
    _, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(np.rad2deg(angles), np.abs(np.sin(angles) * np.cos(angles)+1), 'r', label='Specific solution')
    ax1.plot(np.rad2deg(angles), support, 'b.', markersize=4, label='General solution')
    # ax1.set_xlabel('Angle [deg]')
    ax1.set_ylabel(r'Area [mm$^2$]')
    ax1.legend()

    ax2.plot(np.rad2deg(angles), finite_central_differences(support, angles), 'k.', label='Finite differences')
    ax2.plot(np.rad2deg(angles), np.sin(4 * angles) / np.abs(2 * np.sin(2 * angles)), 'r', label='Specific solution')
    ax2.plot(np.rad2deg(angles), dSdt, 'b.', markersize=6, label=r"General solution")
    ax2.set_xlabel('Angle [deg]')
    ax2.set_ylabel(r'Area derivative [mm$^2$/deg]')
    ax2.legend()

    fig.suptitle('Area below unit square, comparison of general and specific solutions')

    plt.savefig('out/supportvolume/2D/2D_solution_comp_no_proj.svg', format='svg')
    fig.show()

    # compare general solutions, no projection vs adaptive projection
    fig, ax = plt.subplots(1, 1)
    ax.plot(np.rad2deg(angles), support, 'r', label='No projection')
    ax.plot(np.rad2deg(angles), support_adap, 'b.', label='Projection to lowest point')
    ax.legend()
    ax.set_xlabel('Angle [deg]')
    ax.set_ylabel(r'Area [mm$^2$]')
    fig.suptitle('Solution comparison: no projection vs. projection to lowest point')
    plt.savefig('out/supportvolume/2D/2D_comp_no_proj_vs_adap_proj.svg', format='svg')
    fig.show()
