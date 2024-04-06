import time
import pyvista as pv
import numpy as np
from helpers import *


def generate_connectivity(mesh):
    res = []

    # add centroid coordinate to cells
    mesh.cell_data['Center'] = [np.sum(c.points, axis=0) / 3 for c in mesh.cell]

    for i in range(mesh.n_cells):
        arr = []

        for j in range(mesh.n_cells):
            if i == j:
                continue

            # # extract cells and normal vectors
            # ci = mesh.extract_cells(i)
            # cj = mesh.extract_cells(j)

            # check if normals point towards each other
            line = mesh['Center'][j] - mesh['Center'][i]
            line = line/np.linalg.norm(line)
            if np.dot(line, mesh['Normals'][i]) > 0:

                # do ray tracing for check
                _, ids = mesh.ray_trace(mesh['Center'][i], mesh['Center'][i] + 2 * line)

                # check that first intersected cell is j, otherwise append first intersected idx
                ids = ids[ids != i]
                arr.append(ids[0])
        res.append(list(set(arr)))

    return res


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
