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
    # load parameters
    FILENAME = par['Filepath']
    NUM_IT = par['Res angle']
    MAX_ANGLE = par['Max angle']
    ORIGIN = par['Origin']
    proj_origin = [0, 0, -20]

    # create a pv Plotter and show origin
    plot = pv.Plotter()  # type: ignore
    plot.add_axes_at_origin()

    # create a mesh, move the CoG to the origin and add to plotter
    mesh = Mesh(FILENAME)
    _ = mesh.move_cog_to(ORIGIN)
    plot.add_mesh(mesh.data, color='green', name='object')

    # create a projection of the mesh on the xy plane,
    # add to plotter and show plot
    proj = mesh.project_points_to_plane(origin=proj_origin)
    plot.add_mesh(proj, color='blue', name='projection')
    plot.show(interactive_update=True)

    # iterate over angles, and gather projected areas
    ax = ay = np.linspace(-MAX_ANGLE, MAX_ANGLE, NUM_IT)
    f = np.zeros((ax.shape[0], ay.shape[0]))
    rot = Rotation.from_matrix(np.identity(3))
    tfm = np.zeros((4, 4))
    tfm[:-1, :-1] = rot.as_matrix()
    tfm[-1, -1] = 1

    start = time.time()
    for i, y in enumerate(ax):
        for j, x in enumerate(ay):
            time.sleep(0.01)

            # get inverse orientation and transform back to initial config
            inv_or = rot.inv()
            tfm[:-1, :-1] = inv_or.as_matrix()
            mesh.transform(tfm)

            # now do a new rotation based on Euler angles
            rot = Rotation.from_euler('XYZ', [x, y, 0], True)
            tfm[:-1, :-1] = rot.as_matrix()
            mesh.transform(tfm)

            plot.update()

            proj = mesh.project_points_to_plane(origin=proj_origin)
            plot.add_mesh(proj, color='blue', name='projection')
            plot.update()

            area = proj.area
            print(f'Area at x,y: {x, y}: {area}')
            f[i, j] = area

    end = time.time()
    print(f'execution duration: {end-start} seconds')
    # plot.show()

    # surface plot
    x, y = np.meshgrid(ax, ay)
    surface = pv.StructuredGrid(x, y, f)
    surface.plot(show_edges=True, show_grid=True)


if __name__ == "__main__":
    main()
