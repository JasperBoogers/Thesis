from typing import Callable
import numpy as np
import pyvista as pv
from parameters import par
import time


class Mesh:

    def __init__(self, file, x=0, y=0) -> None:
        self.data = pv.read(file)
        self.rot_x = x
        self.rot_y = y

    def get_orientation(self) -> tuple:
        return self.rot_x, self.rot_y

    def __getattr__(self, attr) -> Callable:
        return getattr(self.data, attr)

    def rotate_x(self, angle, point=(0, 0, 0), abs=False) -> None:
        if abs:
            a = angle - self.rot_x
        else:
            a = angle

        self.data.rotate_x(a, point=point, inplace=True)  # type: ignore
        self.rot_x = self.rot_x + a

    def rotate_y(self, angle, point=(0, 0, 0), abs=False) -> None:
        if abs:
            a = angle - self.rot_y
        else:
            a = angle

        self.data.rotate_y(a, point=point, inplace=True)  # type: ignore
        self.rot_y = self.rot_y + a

    def move_to_origin(self) -> None:
        """Moves the center of mass to its origin
        """
        cog = self.data.center_of_mass()  # type: ignore
        self.data.translate(-cog, inplace=True)  # type: ignore


def eval_f(m):
    cog = m.data.center_of_mass()
    return cog[-1]


def main():
    # load parameters
    FILENAME = par['Filepath']
    NUM_IT = par['Res angle']
    MAX_ANGLE = par['Max angle']

    file = pv.read(FILENAME)
    mesh = Mesh(FILENAME)
    mesh.move_to_origin()

    plot = pv.Plotter()  # type: ignore
    plot.add_axes_at_origin()
    plot.add_mesh(mesh.data, color='green')

    plot.show(interactive_update=True)

    # perform rotations
    ax = ay = np.linspace(-MAX_ANGLE, MAX_ANGLE, NUM_IT)
    f = np.zeros((NUM_IT, NUM_IT))

    start = time.time()
    for i, x in enumerate(ax):
        for j, y in enumerate(ay):
            time.sleep(2)
            mesh.rotate_x(x, abs=True)
            mesh.rotate_y(y, abs=True)
            print(mesh.get_orientation())
            plot.update()
            f[i, j] = eval_f(mesh)

    end = time.time()
    plot.show()

    # # surface plot
    # x, y = np.meshgrid(ax, ay)
    # surface = pv.StructuredGrid(x, y, f)
    # surface.plot(show_edges=True, show_grid=True)
    
    print(f'execution duration: {end-start} seconds')


if __name__ == "__main__":
    main()
