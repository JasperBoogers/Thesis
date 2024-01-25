import os
import numpy as np
import pandas as pd
import pyvista as pv
import time


def run_SuperSlicer(slicer: str, geometry: str, config: str, outfolder: str):
    # build command
    cmd = f'{slicer} -g {geometry} -o {outfolder} --load {config}'
    os.system(f'cmd /c {cmd}')


def extract_characteristics(m: pv.DataSet | pv.PolyData) -> tuple[float, float, float]:
    if not m.is_all_triangles:
        m = m.clean().triangulate()

    # (re)calculate metrics
    m = m.compute_cell_sizes()
    _ = m.set_active_scalars('Area')

    volume = m.volume
    area = sum(m.active_scalars)
    height: float = m.bounds[-1] - m.bounds[-2]

    return volume, area, height


def extract_time(filename: str) -> float:
    with open(filename, 'r') as f:
        lines = f.readlines()

        # find line containing print time and strip
        fil = filter(lambda x: 'estimated printing time (normal mode)' in x, lines)
        line = list(fil)[0]
        line = line.strip('\n').split('=')[1]

        # split on space char, strip the first space and extract time
        if 'h' in line:
            hour, minute, sec = [int(l[:-1]) for l in line.split(' ')[1:]]
            tm = 3600 * hour + 60 * minute + sec
        else:
            minute, sec = [int(l[:-1]) for l in line.split(' ')[1:]]
            tm = 60 * minute + sec
    return tm


if __name__ == '__main__':

    # define paths
    CWD = os.getcwd()
    SLICER_PATH = "SuperSlicer\\superslicer_console.exe"
    GEOMETRY_PATH = os.path.join(CWD, "Geometries")
    OUTDIR = './gcodes'
    CONFIG = "config.ini"
    OUTFILE = os.path.join(CWD, "time_estimation.xlsx")
    DF_COLUMNS = ["File name", "Volume [mm3]", "Area [mm2]", "Height [mm]", "Height^2 [mm2]", "Height^3 [mm3]", "Time [s]"]

    # get list of all geometries
    geometries = os.listdir(GEOMETRY_PATH)
    geometries = [g.strip('.stl') for g in geometries]

    # check if OUTFILE exists, if not create an empty df
    if os.path.isfile(OUTFILE):
        print('Found existing file ', OUTFILE)
        df = pd.read_excel(OUTFILE)
    else:
        print('Creating new file ', OUTFILE)
        df = pd.DataFrame(columns=DF_COLUMNS)

    # define lists of data to extract
    files2append = []
    times2append = []
    volumes2append = []
    areas2append = []
    heights2append = []

    # now check if new geometries have been added to Geometries dir
    existing_geometries = list(df['File name'])
    new_geometries = [file for file in geometries if file not in existing_geometries]

    # empty the out folder
    for file in os.listdir(OUTDIR):
        os.remove(os.path.join(OUTDIR, file))

    start = time.time()

    print(f'Generating G-code for {len(new_geometries)} files')
    for file in new_geometries:
        fn = f'./Geometries/{file}.stl'
        run_SuperSlicer(SLICER_PATH, fn, CONFIG, OUTDIR)

    # extract time and characteristics from new files
    for file in os.listdir(OUTDIR):
        # get time
        t = extract_time(os.path.join(OUTDIR, file))

        # now extract characteristics for file
        mesh = pv.read(os.path.join(GEOMETRY_PATH, file.strip('.gcode') + '.stl'))
        v, a, h = extract_characteristics(mesh)

        # append data
        files2append.append(file.strip('.gcode'))
        times2append.append(t)
        volumes2append.append(v)
        areas2append.append(a)
        heights2append.append(h)

    print(f'Extraction took {time.time() - start} seconds')

    # construct new df and concat with existing df
    data = {'File name': files2append, 'Time [s]': times2append, 'Volume [mm3]': volumes2append, 'Area [mm2]': areas2append, 'Height [mm]': heights2append}
    for col in DF_COLUMNS:
        if col not in ['File name', 'Time [s]', 'Volume [mm3]', 'Area [mm2]', 'Height [mm]']:
            data[col] = len(files2append)*[np.nan]

    new_df = pd.DataFrame(data)
    new_df = pd.concat([df, new_df], axis=0, ignore_index=True)

    # add extra polynomials
    new_df['Height^2 [mm2]'] = new_df['Height [mm]']**2
    new_df['Height^3 [mm3]'] = new_df['Height [mm]']**3

    # save df
    new_df.to_excel('time_estimation_new.xlsx', index=False)

    print('finished')
