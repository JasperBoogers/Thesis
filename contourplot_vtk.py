import numpy as np
import time
import vtkmodules.vtkInteractionStyle
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer
)
from vtkmodules.vtkIOGeometry import vtkSTLReader
import os
from parameters import par


# make evaluation function
def eval(model):
    z = 1
    return z


def main():
    # load parameters
    FILENAME = par['Filepath']
    NUM_IT = par['Res angle']
    MAX_ANGLE = par['Max angle']

    # set up vtk actor
    reader = vtkSTLReader()
    reader.SetFileName(FILENAME)

    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(reader.GetOutputPort())

    actor = vtkActor()
    actor.SetMapper(mapper)

    ren = vtkRenderer()
    renWin = vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetWindowName('ReadSTL')
    renWin.SetSize(300, 300)

    iren = vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    ren.AddActor(actor)

    iren.Initialize()
    ren.ResetCamera()
    ren.GetActiveCamera().Zoom(1.5)

    renWin.Render()
    iren.Start()


if __name__ == "__main__":
    main()

# # perform rotations
# ax = ay = np.linspace(-MAX_ANGLE, MAX_ANGLE, NUM_IT)
# f = np.zeros((NUM_IT, NUM_IT))

# start = time.time()
# for i, t in enumerate(ax):
#     for j, p in enumerate(ay):
#         # copy obj in base orientation

#         # orient about x/y axis

#         f[i, j] = eval(m)
# end = time.time()
# print(f'execution duration: {end-start} seconds')

# plot
