from vtk import vtkTransform
# from scipy.spatial.transform import Rotation as r
import numpy as np
import pyvista as pv
from Mesh import Mesh
from parameters import par

FILE = par['Filepath']
ORIGIN = [0, 0, 0]
mesh = Mesh(FILE)
_ = mesh.move_cog_to(ORIGIN)

# create a pv Plotter and show origin
plot = pv.Plotter()  # type: ignore
plot.add_axes_at_origin()

# create a mesh, move the CoG to the origin and add to plotter
plot.add_mesh(mesh.data, color='green', name='object')

# show results of vtkTransform
print('Rotate without Identity()')
tfm = vtkTransform()
tfm.PostMultiply()
tfm.RotateX(-90)
print(f'rotated -90 degrees: {tfm.GetOrientation()}')
print(f'Nr. of concatenated transforms: {tfm.GetNumberOfConcatenatedTransforms()}')
tfm.RotateX(90)
print(f'rotated 90 degrees: {tfm.GetOrientation()}')
print(f'Nr. of concatenated transforms: {tfm.GetNumberOfConcatenatedTransforms()}')

print('Now use Identity() to reset transform')
tfm_id = vtkTransform()
tfm_id.PostMultiply
tfm_id.RotateX(-90)
print(f'rotated -90 degrees: {tfm_id.GetOrientation()}')
print(f'Nr. of concatenated transforms: {tfm_id.GetNumberOfConcatenatedTransforms()}')
tfm_id.Identity()
print(f'Used Identity(): {tfm_id.GetOrientation()}')
print(f'Nr. of concatenated transforms: {tfm_id.GetNumberOfConcatenatedTransforms()}')
