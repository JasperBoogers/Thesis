import time
import pyvista as pv
import numpy as np
from helpers import *





if __name__ == '__main__':

    # load file
    FILE = 'Geometries/cube_cutout.stl'
    m = pv.read(FILE)
    m = prep_mesh(m, decimation=0, translate=True)
    m = m.subdivide(2, subfilter='linear')

    start = time.time()
    connectivity = generate_connectivity(m)
    end = time.time()
    print('Time: ', end - start)

    i = 100
    p = pv.Plotter()
    _ = p.add_mesh(m, style='wireframe', color='k')
    _ = p.add_mesh(m.extract_cells(i), color='g')
    if len(connectivity[i]) > 0:
        _ = p.add_mesh(m.extract_cells(connectivity[i]), color='b')
    p.show()
