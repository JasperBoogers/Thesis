import numpy as np
import pyvista as pv
from parameters import par
import time
from Mesh import Mesh
# from vtk import vtkTransform
from scipy.spatial.transform import Rotation


def eval_f(model: Mesh, plane):
    proj = model.project_points_to_plane(origin=plane)
    return proj.area


def main():
    # set parameters
    FILENAME = par['Filepath']
    NUM_IT = par['Res angle']
    MAX_ANGLE = par['Max angle']
    ORIGIN = par['Origin']
    PROJ_OFFSET = -5
    REDUCTION = par['Resolution reduction']
    OUTPATH = par['Outpath']
    gif = False
    vtp = False

    # create a pv Plotter and show origin
    plot = pv.Plotter()  # type: ignore
    # plot.add_axes_at_origin()
    plot.add_axes(shaft_length=0.9)

    # create a mesh
    mesh = Mesh(FILENAME)
    # move the "center" to the origin, decimate, and add to plotter
    mesh.decimate_pro(REDUCTION)
    _ = mesh.move_center_to(ORIGIN)
    plot.add_mesh(mesh.data, show_edges=True, color='green', name='object')

    # create a projection of the mesh on the xy plane,
    # add to plotter and show plot
    proj_origin = [0, 0, mesh.bounds[-2] + PROJ_OFFSET]  # type: ignore
    proj = mesh.project_points_to_plane(origin=proj_origin)
    plot.add_mesh(proj, color='blue', name='projection')
    plot.show(interactive_update=True)
    if gif:
        plot.open_gif('benchy.gif')

    # iterate over angles, and gather projected areas
    ax = ay = np.linspace(-MAX_ANGLE, MAX_ANGLE, NUM_IT)
    f = np.zeros((ax.shape[0], ay.shape[0]))
    rot = Rotation.from_matrix(np.identity(3))
    tfm = np.identity(4)

    # parameters for output generation
    meshes = []

    start = time.time()
    for i, x in enumerate(ax):
        for j, y in enumerate(ay):

            # get inverse orientation and transform back to initial config
            inv_or = rot.inv()
            tfm[:-1, :-1] = inv_or.as_matrix()
            mesh.transform(tfm)

            # now do a new rotation based on Euler angles
            rot = Rotation.from_euler('XYZ', [x, y, 0], True)
            tfm[:-1, :-1] = rot.as_matrix()
            mesh.transform(tfm)

            # project the mesh to the z plane
            proj_origin = [0, 0, mesh.bounds[-2] + PROJ_OFFSET]  # type: ignore
            proj = mesh.project_points_to_plane(origin=proj_origin)
            plot.add_mesh(proj, color='blue', name='projection')

            # update the plot and write to file
            plot.update()
            if gif:
                plot.write_frame()
            meshes.append(mesh.data.copy())  # type: ignore

            area = proj.area
            f[i, j] = area

    if vtp:
        for n, m in enumerate(meshes):
            filename = OUTPATH + f'3DBenchy-{n}.vtp'
            m.save(filename)

    # retrieve the optimal orientation
    opt_idx = np.unravel_index(np.argmin(f), f.shape)
    x = ax[opt_idx[0]]
    y = ay[opt_idx[1]]
    print(f'optimal orientation at {x, y} degrees, f={f[opt_idx]}')

    if gif:
        plot.close()
    else:
        # get inverse orientation and transform back to initial config
        inv_or = rot.inv()
        tfm[:-1, :-1] = inv_or.as_matrix()
        mesh.transform(tfm)

        # now do a new rotation based on Euler angles
        rot = Rotation.from_euler('XYZ', [x, y, 0], True)
        tfm[:-1, :-1] = rot.as_matrix()
        mesh.transform(tfm)

        # project the mesh to the z plane
        proj_origin = [0, 0, mesh.bounds[-2] + PROJ_OFFSET]  # type: ignore
        proj = mesh.project_points_to_plane(origin=proj_origin)
        plot.add_mesh(proj, color='blue', name='projection')
        plot.update()

    end = time.time()
    print(f'execution duration: {end-start} seconds')

    # surface plot
    x, y = np.meshgrid(ax, ay, indexing='ij')
    surface = pv.StructuredGrid(x, y, f)
    surf_plot = pv.Plotter()  # type: ignore
    surf_plot.add_mesh(surface, scalars=surface.points[:, -1], show_edges=True,
                       scalar_bar_args={'vertical': True})
    surf_plot.set_scale(zscale=0.1)
    surf_plot.add_title('3DBency - projected area')
    surf_plot.show_grid()
    surf_plot.show()


if __name__ == "__main__":
    main()
