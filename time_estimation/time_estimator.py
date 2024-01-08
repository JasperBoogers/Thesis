import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FILE = 'time_estimation.xlsx'
df = pd.read_excel(FILE)

A = df.drop(['Filename', 'Time [s]'], axis=1).to_numpy()
b = df['Time [s]'].to_numpy()
x = np.linalg.lstsq(A, b, rcond=None)[0]

n = 3

_ = plt.plot(A[:, n], b, 'o', label='Original')
_ = plt.plot(A[:, n], x[n]*A[:, n], 'r', label='Fitted line')
_ = plt.legend()
plt.show()
