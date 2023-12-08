import os
import numpy as np


par = {
    'Filepath': os.path.join('Geometries', '3DBenchy', '3DBenchy.stl'),
    'Max angle': 90,
    'Res angle': 5,
    'Origin': np.array([0, 0, 0]),
    'Resolution reduction': 0.9,
    'Outpath': 'out/contourplot/3DBenchy/'
}
