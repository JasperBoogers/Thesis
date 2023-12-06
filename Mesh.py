from typing import Callable
import pyvista as pv
import numpy as np
from vtk import vtkTransform


class Mesh:

    def __init__(self, file, x=0, y=0) -> None:
        self.data = pv.read(file)
        self.tfm = vtkTransform()

        # (re)set rotation matrix
        self.tfm.Identity()
        self.tfm.PostMultiply()
        self.tfm.RotateX(x)
        self.tfm.RotateY(y)

    def get_orientation(self) -> tuple:
        return self.tfm.GetOrientation()

    def __getattr__(self, attr) -> Callable:
        return getattr(self.data, attr)

    def rotate_x(self, angle, abs=False) -> None:
        if abs:
            o = self.get_orientation()

            # reset orientation
            self.tfm.Identity()
            self.tfm.RotateX(angle)
            self.tfm.RotateY(o[1])
        else:
            self.tfm.RotateX(angle)
        self.data.transform(self.tfm, inplace=True)  # type: ignore

    def rotate_y(self, angle, abs=False) -> None:
        if abs:
            o = self.get_orientation()

            # reset orientation
            self.tfm.Identity()
            self.tfm.RotateX(o[0])
            self.tfm.RotateY(angle)
        else:
            self.tfm.RotateY(angle)

        self.data.transform(self.tfm, inplace=True)  # type: ignore

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
