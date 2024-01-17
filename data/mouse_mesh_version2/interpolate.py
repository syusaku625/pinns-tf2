import numpy as np
import pandas as pd
from scipy import interpolate

loop = 9


filename = "tmp_contrast0.csv"
df = pd.read_csv(filename)
tmp_value = df['concentration[-]'].to_numpy()
data_length = len(tmp_value)

contrast_value = [[] for i in range(loop)]


for i in range(0, loop):
    filename = "tmp_contrast" + str(i) + ".csv"
    df = pd.read_csv(filename)
    tmp_contrast = df['concentration[-]'].to_numpy()
    #print(tmp_contrast)
    for j in tmp_contrast:
        contrast_value[i].append(float(j))


contrast_value = np.array(contrast_value).T

x = np.arange(0, loop)
xnew = np.arange(0, 8.1, 0.1)

contrast_value_interpolate = [[] for i in range(data_length)]

count = 0

for i in contrast_value:
    f = interpolate.interp1d(x, i)
    contrast_new = f(xnew)
    contrast_value_interpolate[count] = contrast_new
    count += 1

contrast_value_interpolate = np.array(contrast_value_interpolate).T

for count, i in enumerate(contrast_value_interpolate):
    output_filename = 'contrast_interpolated_' + str(count) + '.csv'
    df = pd.DataFrame(i)
    name = ['concentration[-]']
    df.to_csv(output_filename, header = name, index = None)

    