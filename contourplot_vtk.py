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
    vtkCubeSource,
    vtkCenterOfMass)
import pyvista as pv
from parameters import par


# make evaluation function
def eval_f(model):
    flt = vtkCenterOfMass()
    flt.SetInputData(model)
    flt.SetUseScalarsAsWeights(False)
    flt.Update()
    cog = flt.GetCenter()
    return cog[-1]


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

    # load stl file
    reader = vtkSTLReader()
    reader.SetFileName(FILENAME)
    reader.Update()

    tfm = vtkTransform()
    tfm.Identity()

    filt = vtkTransformPolyDataFilter()
    # filt.SetInputConnection(reader.GetOutputPort())
    filt.SetInputData(move_to_origin(reader.GetOutput()))
    filt.SetTransform(tfm)

    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(filt.GetOutputPort())
    actor = vtkActor()
    actor.SetMapper(mapper)

    cube = vtkCubeSource()
    cube.SetCenter(0, 0, 0)
    cube.SetYLength(2)
    cubemap = vtkPolyDataMapper()
    cubemap.SetInputConnection(cube.GetOutputPort())
    cubeActor = vtkActor()
    cubeActor.SetMapper(cubemap)

    # plot
    ren = vtkRenderer()
    ren.AddActor(actor)
    ren.AddActor(cubeActor)
    renWin = vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetWindowName('ReadSTL')
    renWin.SetSize(300, 300)

    iren = vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    # add origin
    widget = vtkOrientationMarkerWidget()
    rgba = [0] * 4
    vtkNamedColors().GetColor('Carrot', rgba)
    widget.SetOutlineColor(rgba[0], rgba[1], rgba[2])
    widget.SetOrientationMarker(vtkAxesActor())
    widget.SetInteractor(iren)
    widget.SetEnabled(1)

    iren.Initialize()
    ren.ResetCamera()
    ren.GetActiveCamera().Zoom(0.5)

    # perform rotations
    ax = ay = np.linspace(-MAX_ANGLE, MAX_ANGLE, NUM_IT)
    f = np.zeros((NUM_IT, NUM_IT))

    start = time.time()
    for i, x in enumerate(ax):
        for j, y in enumerate(ay):
            # create new transform instance
            time.sleep(0.005)
            tfm.Identity()
            tfm.RotateX(x)
            tfm.RotateY(y)

            print(tfm.GetOrientation())

            # update pipeline
            renWin.Render()

            f[i, j] = eval_f(filt.GetOutput())

    end = time.time()
    print(f'execution duration: {end-start} seconds')
    # iren.Start()

    # surface plot
    x, y = np.meshgrid(ax, ay)
    surface = pv.StructuredGrid(x, y, f)
    surface.plot(show_edges=True, show_grid=True)


if __name__ == "__main__":
    main()
