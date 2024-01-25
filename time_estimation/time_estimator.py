import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FILE = 'time_estimation.xlsx'
df = pd.read_excel(FILE)
print(df)

A = df.drop(['File name', 'Time [s]'], axis=1).to_numpy()
b = df['Time [s]'].to_numpy()
x = np.linalg.lstsq(A, b, rcond=None)[0]
print(x)

n = 2
y = x[n]*A[:, n] + x[n+1]*A[:, n] + x[n+2]*A[:, n]

_ = plt.plot(A[:, n], b, 'o', label='Original')
_ = plt.plot(A[:, n], y, 'r', label='Fitted line')
_ = plt.legend()
plt.show()
