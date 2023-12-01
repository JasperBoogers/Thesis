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
from vtk import (
    vtkTransform,
    vtkTransformPolyDataFilter,
    vtkOrientationMarkerWidget,
    vtkNamedColors,
    vtkAxesActor,
    vtkCenterOfMass)
import os
from parameters import par


# make evaluation function
def eval(model):
    return 1


def move_to_origin(obj):
    flt = vtkCenterOfMass()
    flt.SetInputData(obj)
    flt.Update()
    cog = flt.GetCenter()

    tfm = vtkTransform()
    tfm.Translate(-cog[0], -cog[1], -cog[2])
    tfm.Update()

    flt = vtkTransformPolyDataFilter()
    flt.SetInputData(obj)
    flt.SetTransform(tfm)
    flt.Update()

    return flt.GetOutput()


def rotate(model, x, y):
    tf = vtkTransform()
    tf.RotateWXYZ(x, 1, 0, 0)
    tf.RotateWXYZ(y, 0, 1, 0)

    flt = vtkTransformPolyDataFilter()
    flt.SetInputData(model)
    flt.SetTransform(tf)
    flt.Update()

    return flt.GetOutput()

def main():
    # load parameters
    FILENAME = par['Filepath']
    NUM_IT = par['Res angle']
    MAX_ANGLE = par['Max angle']

    # set up vtk actor
    reader = vtkSTLReader()
    reader.SetFileName(FILENAME)
    reader.Update()
    obj = reader.GetOutput()

    mapper = vtkPolyDataMapper()
    actor = vtkActor()
    actor.SetMapper(mapper)

    # plot
    ren = vtkRenderer()
    renWin = vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetWindowName('ReadSTL')
    renWin.SetSize(300, 300)

    iren = vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    ren.AddActor(actor)

    # add origin
    widget = vtkOrientationMarkerWidget()
    rgba = [0] * 4
    vtkNamedColors().GetColor('Carrot', rgba)
    widget.SetOutlineColor(rgba[0], rgba[1], rgba[2])
    widget.SetOrientationMarker(vtkAxesActor())
    widget.SetInteractor(iren)
    widget.SetViewport(0.0, 0.0, 0.4, 0.4)
    widget.SetEnabled(1)
    widget.InteractiveOn()

    iren.Initialize()

    # move obj to origin
    obj = move_to_origin(obj)
    ren.ResetCamera()

    # perform rotations
    ax = ay = np.linspace(-MAX_ANGLE, MAX_ANGLE, NUM_IT)
    f = np.zeros((NUM_IT, NUM_IT))

    mapper.SetInputData(obj)

    start = time.time()
    for i, x in enumerate(ax):
        for j, y in enumerate(ay):
            # orient about x/y axis
            obj_rot = rotate(obj, x, y)
            mapper.SetInputData(obj_rot)

            ren.ResetCamera()
            renWin.Render()
            time.sleep(0.01)

            f[i, j] = eval(obj_rot)

    end = time.time()
    print(f'execution duration: {end-start} seconds')

 
if __name__ == "__main__":
    main()
