import pyvista as pv
import numpy as np
import pandas as pd
from os import path, getcwd, listdir
from datetime import datetime as dt


def extract_characteristics(mesh: pv.DataSet | pv.PolyData) -> tuple[float, float, float]:

    if not mesh.is_all_triangles:
        mesh = mesh.clean().triangulate()

    # (re)calculate metrics
    mesh = mesh.compute_cell_sizes()
    _ = mesh.set_active_scalars('Area')

    volume = mesh.volume
    area = sum(mesh.active_scalars)
    height: float = mesh.bounds[-1] - mesh.bounds[-2]

    return volume, area, height


if __name__ == '__main__':
    cwd = getcwd()
    PATH = path.join(cwd, 'Geometries')
    files = listdir(PATH)
    OUTFILE = 'time_estimation.xlsx'

    # check if OUTFILE exists, if not create an empty df
    if path.isfile(path.join(cwd, OUTFILE)):
        print('Found existing file ', OUTFILE)
        df = pd.read_excel(OUTFILE)
    else:
        df = pd.DataFrame()

    # now check if new geometries have been added to Geometries dir
    existing_geometries = list(df['File name'])
    new_files = [file for file in files if file not in existing_geometries]

    if len(new_files) > 0:
        print(f'Found {len(new_files)} new files')

        volumes = []
        areas = []
        heights = []

        for f in new_files:
            m = pv.read(path.join(PATH, f))
            v, a, h = extract_characteristics(m)
            volumes.append(v)
            areas.append(a)
            heights.append(h)

        assert len(volumes) == len(new_files)

        # construct DataFrame
        data = {'File name': new_files, 'Volume [mm3]': volumes, 'Area [mm2]': areas, 'Height [mm]': heights}
        new_df = pd.DataFrame(data)

        # add extra polynomials
        new_df['Height^2 [mm2]'] = new_df['Height [mm]']**2
        new_df['Height^3 [mm3]'] = new_df['Height [mm]']**3

        df = pd.concat([df, new_df], axis=0)
        df.to_excel(OUTFILE, index=False)
    else:
        print('No new files')
