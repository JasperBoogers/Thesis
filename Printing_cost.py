import pyvista as pv
import numpy as np
from helpers import *
from sensitivities import calc_cell_sensitivities

if __name__ == '__main__':
    # load file
    FILE = 'Geometries/bunny/bunny_coarse.stl'
    m = pv.read(FILE)
    # m = m.subdivide(2, subfilter='linear')
    m = prep_mesh(m, decimation=0)
    connectivity = read_connectivity_csv('out/sim_data/bunny_coarse_connectivity.csv')

    par = {
        'connectivity': connectivity,
        'build_dir': np.array([0, 0, 1]),
        'down_thresh': np.sin(np.deg2rad(45)),
        'up_thresh': np.sin(np.deg2rad(0)),
        'down_k': 10,
        'up_k': 20,
        'plane_offset': calc_min_projection_distance(m),
        'SoP_penalty': 1
    }

    m = calc_cell_sensitivities(m, [3.32341069, -4.07440287], par)

    p = pv.Plotter(off_screen=True)
    _ = p.add_mesh(m, lighting=False, scalars='V', cmap='RdYlGn', clim=[-1, 1], show_edges=True)
    _ = p.add_mesh(pv.Plane(center=[0, 0, m.bounds[-2]], i_size=10, j_size=10), lighting=False, color='k',
                   opacity=0.8)
    p.camera.zoom(1.5)
    p.camera.position = (0, 10, 0)
    p.add_axes()
    path = p.generate_orbital_path(n_points=100, viewup=[0, 0, 1], shift=m.length / 4)
    p.open_gif("out/case_study/Bunny_coarse_cost.gif")
    p.show(auto_close=False)
    p.orbit_on_path(path, viewup=[0, 0, 1], write_frames=True)
    p.close()
