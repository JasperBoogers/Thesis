from vtk import vtkTransform, vtkTransformPolyDataFilter, vtkCenterOfMass, vtkPolyDataNormals, vtkCellCenters
import numpy as np


def vtk_move_to_origin(mesh):

    # compute CoM
    tmp = vtkCenterOfMass()
    tmp.SetInputData(mesh)
    tmp.SetUseScalarsAsWeights(False)
    tmp.Update()
    cen_0 = tmp.GetCenter()

    # translate to origin
    tfm_0 = vtkTransform()
    tfm_0.Translate(-cen_0[0], -cen_0[1], -cen_0[2])
    tfm_0.Update()
    flt = vtkTransformPolyDataFilter()
    flt.SetInputData(mesh)
    flt.SetTransform(tfm_0)
    flt.Update()
    return flt.GetOutput()


def vtk_compute_cell_centers(mesh):
    tmp = vtkCellCenters()
    tmp.SetInputData(mesh)
    tmp.Update()
    ply_tmp = tmp.GetOutput()
    return np.array(ply_tmp.GetPoints().GetData())


def vtk_compute_normals(mesh):
    flt = vtkPolyDataNormals()
    flt.SetInputData(mesh)
    flt.ComputeCellNormalsOn()
    flt.Update()
    return flt.GetOutput()
