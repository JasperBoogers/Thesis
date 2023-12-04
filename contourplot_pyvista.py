import numpy as np
import pyvista as pv
from parameters import par
import time


class Mesh:

    def __init__(self, file, x=0, y=0):
        self.data = pv.read(file)
        self.rot_x = x
        self.rot_y = y

    def get_orientation(self):
        return self.rot_x, self.rot_y

    def __getattr__(self, attr):
        return getattr(self.data, attr)

    def rotate_x(self, angle, point=(0, 0, 0), abs=False):
        if abs:
            a = angle - self.rot_x
        else:
            a = angle

        self.data.rotate_x(a, point=point, inplace=True)
        self.rot_x = self.rot_x + a

    def rotate_y(self, angle, point=(0, 0, 0), abs=False):
        if abs:
            a = angle - self.rot_x
        else:
            a = angle
        
        self.data.rotate_y(a, point=point, inplace=True)
        self.rot_y = self.rot_y + a

    def move_to_origin(self):
        cog = self.data.center_of_mass()
        self.data.translate(-cog, inplace=True)


    # def set_rotation(self, x, y, inplace=True):
    #     super().rotate_x(x - self.rot_x, inplace=inplace)
    #     super().rotate_y(y - self.rot_y, inplace=inplace)
    #     self.rot_x = x
    #     self.rot_y = y


def main():
    # load parameters
    FILENAME = par['Filepath']
    NUM_IT = par['Res angle']   
    MAX_ANGLE = par['Max angle']

    file = pv.read(FILENAME)
    mesh = Mesh(FILENAME)

    # add origin
    ax = pv.Axes(show_actor=True)
    ax.origin = (0, 0, 0)

    plot = pv.Plotter()
    plot.add_actor(ax.actor)
    plot.add_mesh(mesh.data, color='green')

    plot.show(interactive_update=True)

    # perform rotations
    ax = ay = np.linspace(-MAX_ANGLE, MAX_ANGLE, NUM_IT)
    f = np.zeros((NUM_IT, NUM_IT))

    start = time.time()
    time.sleep(2)
    mesh.move_to_origin()
    plot.update()
    time.sleep(2)
    mesh.rotate_x(90)
    plot.update()
    time.sleep(2)
    mesh.rotate_x(-90)
    plot.update()

    plot.show()

    end = time.time()
    print(f'execution duration: {end-start} seconds')

if __name__ == "__main__":
    main()
