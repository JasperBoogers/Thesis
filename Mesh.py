from typing import Callable
import pyvista as pv
import numpy as np


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

    def move_cog_to(self, point=[0, 0, 0]) -> np.ndarray:
        """Moves the center of mass to its origin
        """
        if type(point) is not np.ndarray:
            point = np.array(point)

        cog = self.data.center_of_mass()  # type: ignore
        loc = point - cog
        self.move_to(loc)
        return cog

    def move_to(self, p) -> None:
        self.data.translate(p, inplace=True)  # type: ignore
