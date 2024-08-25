import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# parameters
TRAIN_RATIO = 1
SAVE = False
FILE = 'time_estimation.xlsx'
df = pd.read_excel(FILE)

# convert units from mm to m to prevent rank deficiency later
df['Volume [mm3]'] /= 1000
df['Area [mm2]'] /= 1000
df['Height [mm]'] /= 1000

# add extra polynomial terms
df['Volume**2'] = df['Volume [mm3]']**2
df['Volume**3'] = df['Volume [mm3]']**3
df['Area**2'] = df['Area [mm2]']**2
df['Area**3'] = df['Area [mm2]']**3
df['Height**2'] = df['Height [mm]']**2
df['Height**3'] = df['Height [mm]']**3

# distribute training and validation data according to train/validation ratio
training_data = df
validation_data = df

# select training data
A = training_data.drop(['File name', 'Time [s]'], axis=1).to_numpy()
b = training_data['Time [s]'].to_numpy()

# select validation data
validation_x = validation_data.drop(['File name', 'Time [s]'], axis=1).to_numpy()
validation_y = validation_data['Time [s]'].to_numpy()

# make RoM using LLSQ
LLSQ = np.linalg.lstsq(A, b, rcond=None)
x = LLSQ[0]
print(f'Sum of residuals: {LLSQ[1]}')
model_y = validation_x @ x

# plot
fig = plt.figure()
_ = plt.plot(validation_y, model_y, 'r.', label='Fitted line')
_ = plt.axline((0, 0), slope=1)
plt.xlabel('Slicer time [s]')
plt.ylabel('Estimated time [s]')
plt.title('Slicer vs. estimated time')
if SAVE:
    plt.savefig('estimation_fit.svg', format='svg', bbox_inches='tight')
plt.show()

fig = plt.figure()
_ = plt.plot(model_y, model_y-validation_y, '.', label='Residuals')
plt.xlabel('Estimated time [s]')
plt.ylabel('Residual [s]')
plt.title('Residuals over estimated time ')
if SAVE:
    plt.savefig('estimation_residuals.svg', format='svg', bbox_inches='tight')
plt.show()
