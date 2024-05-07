import pyvista as pv
import numpy as np
from helpers import *
from sensitivities import calc_cell_sensitivities

if __name__ == '__main__':
    # load file
    FILE = 'Geometries/jet_bracket_2.stl'
    m = pv.read(FILE)
    # m = m.subdivide(2, subfilter='linear')
    m = prep_mesh(m, decimation=0.8)
    connectivity = read_connectivity_csv('out/sim_data/jet_bracket_2_conn.csv')
    assert len(connectivity) == m.n_cells

    args = {
        'connectivity': connectivity,
        'build_dir': np.array([0, 0, 1]),
        'down_thresh': np.sin(np.deg2rad(45)),
        'up_thresh': np.sin(np.deg2rad(0)),
        'down_k': 10,
        'up_k': 10,
        'SoP_penalty': 1,
        'softmin_p': -15
    }

    m = calc_cell_sensitivities(m, np.deg2rad([481.92160597, 179.79582416]), args)
    bounds = m.bounds
    plane = pv.Plane(center=(0, 0, bounds[-2]), i_size=1.1 * (bounds[1] - bounds[0]),
                     j_size=1.1 * (bounds[3] - bounds[2]), direction=(0, 0, 1))

    p = pv.Plotter(off_screen=True)
    _ = p.add_mesh(m, lighting=False, scalars='V', cmap='RdYlBu', clim=[-1, 1], show_edges=True)
    _ = p.add_mesh(plane, lighting=False, color='k', opacity=0.8)
    p.camera.zoom(1.5)
    p.camera.position = (0, 10, 0)
    p.add_axes()
    path = p.generate_orbital_path(n_points=100, viewup=[0, 0, 1], shift=m.length / 4)
    p.open_gif("out/case_study/Jet_bracket_2_cost.gif")
    p.show(auto_close=False)
    p.orbit_on_path(path, viewup=[0, 0, 1], write_frames=True)
    p.close()
