import pandas as pd
import numpy as np
import csv


def read_connectivity_csv(filename):
    # df = pd.read_csv(filename, header=None)
    # data = df.values
    #
    # connectivity = []
    # for line in data:
    #     if line[0] == -1:
    #         # this idx has no connectivity, add empty list
    #         connectivity.append([])
    #     else:
    #         # remove nan values
    #         line = line[~np.isnan(line)]
    #         # add list of integers to connectivity array
    #         connectivity.append(list(line.astype(int)))

    with open(filename, 'r') as file:
        data = [row for row in csv.reader(file)]

    df = pd.DataFrame(data)
    if df.shape[-1] == 1:
        df = df[0].str.split(',', expand=True)
    data = df.values.astype(float)

    connectivity = []
    for line in data:
        if line[0] == -1:
            # this idx has no connectivity, add empty list
            connectivity.append([])
        else:
            # remove nan values
            line = line[~np.isnan(line)]
            # add list of integers to connectivity array
            connectivity.append(list(line.astype(int)))

    return connectivity


def write_connectivity_csv(connectivity, filename):
    # empty lists get a -1 to ensure their entry in the csv
    conn = [c if len(c) > 0 else [-1] for c in connectivity]

    write_csv(conn, filename)


def read_csv(filename, sep=',', dtype=float):
    df = pd.read_csv(filename, sep=sep, header=None)
    return df.astype(dtype).values


def write_csv(data, filename):

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)
