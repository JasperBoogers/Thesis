import pyvista as pv
import numpy as np
from helpers import *

if __name__ == '__main__':
    cube = pv.read('Geometries/chair.stl')
    # cube = pv.Cube()
    cube = prep_mesh(cube, decimation=0)
    cube = cube.subdivide(2, 'linear')

    z_min = np.array([0, 0, -calc_min_projection_distance(cube)])
    b = np.array([0, 0, 1])
    a = [0, np.deg2rad(45)]
    cube = calc_cell_sensitivities(cube, a, b, z_min)

    p = pv.Plotter(off_screen=True)
    _ = p.add_mesh(cube, lighting=False, scalars='V', cmap='RdYlGn', show_edges=True)
    _ = p.add_mesh(pv.Plane(center=[0, 0, cube.bounds[-2]], i_size=100, j_size=100), lighting=False, color='k',
                   opacity=0.8)
    p.camera.zoom(0.8)
    p.camera.position = (0, 150, 0)
    p.add_axes()
    path = p.generate_orbital_path(n_points=100, viewup=[0, 0, 1], shift=cube.length / 4)
    p.open_gif("out/supportvolume/Cost.gif")
    p.show(auto_close=False)
    p.orbit_on_path(path, viewup=[0, 0, 1], write_frames=True)
    p.close()
