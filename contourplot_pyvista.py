import numpy as np
import pyvista as pv
from parameters import par
import time


def main():
    # load parameters
    FILENAME = par['Filepath']
    NUM_IT = par['Res angle']
    MAX_ANGLE = par['Max angle']

    mesh = pv.read(FILENAME)

    # add origin
    ax = pv.Axes(show_actor=True)
    ax.origin = (0, 0, 0)

    plot = pv.Plotter()
    plot.add_actor(ax.actor)
    plot.add_mesh(mesh)

    plot.show(interactive_update=True)
    time.sleep(2)
    mesh.rotate_x(90, inplace=True)
    plot.update()
    plot.show()


if __name__ == "__main__":
    main()