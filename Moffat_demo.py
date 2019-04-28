import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting

from plot3d import show_3d_data


inp = np.array([[ 2100,  2686,  7445, 10471,  8575,  4416,  3131,],
 [ 2385,  3048,  9470, 14519, 15079, 11714,  9206,]               ,
 [ 2577,  3680, 12177, 22247, 21299, 18572, 12814,]               ,
 [ 2522,  4109, 16691, 31589, 21244, 12947,  6122,]               ,
 [ 2272,  3034, 10765, 16278, 10588,  3978,  2457,]               ,
 [ 2156,  2325,  2886,  3320,  2900,  2420,  2230,]               ,
 [ 2087,  2167,  2262,  2261,  2268,  2197,  2133,]])
# inp = np.array([[ 961,  894,  936,  991,  932,  939,  910,  956,  955,  910,  952,  953,  988, 1006,  970, 1009,  915],
#  [1286, 1259, 1221, 1234, 1161, 1188, 1116, 1228, 1297, 1207, 1286, 1282,  1309, 1291, 1233, 1246, 1208],
#  [1781, 1805, 1651, 1569, 1554, 1504, 1521, 1687, 1785, 1848, 1760, 1707,  1685, 1670, 1659, 1647, 1611],
#  [2070, 2088, 1974, 1901, 1880, 2023, 2019, 2189, 2349, 2082, 2128, 2083,  2091, 1982, 1898, 1892, 1811],
#  [2026, 2036, 2153, 2154, 1955, 2318, 2017, 2344, 2137, 1923, 2066, 2144,  2041, 1915, 1908, 1826, 1786],
#  [1560, 1619, 1854, 1702, 1476, 1462, 1371, 1513, 1319, 1199, 1346, 1453,  1304, 1234, 1375, 1421, 1508]])
print(inp)
# Generate fake data
np.random.seed(0)
x = np.arange(0, inp.shape[1], 1)
y = np.arange(0, inp.shape[0], 1)
matrix_x, matrix_y = np.meshgrid(x, y)
amp_init = np.matrix(inp).max()
halfsize = 5
stdev_init = 0.33 * halfsize

# Fit the data using a box model.
# Bounds are not really needed but included here to demonstrate usage.

def tie_stddev(model):  # we need this for tying x_std and y_std
            xstddev = model.x_stddev
            return xstddev

t_init = models.Gaussian2D(x_mean=halfsize + 0.5, y_mean=halfsize + 0.5, x_stddev=stdev_init,
                                   y_stddev=stdev_init, amplitude=amp_init, tied={'y_stddev': tie_stddev})

# m_init = models.Moffat2D(amplitude=amp_init, x_0=np.argmax(inp)%7, y_0=np.argmax(inp)%7)
fit_m = fitting.LevMarLSQFitter()
m = fit_m(t_init, matrix_x, matrix_y, inp)
print(fit_m.fit_info['message'])

predicted = np.zeros(inp.shape, dtype=int)
for y, row in enumerate(inp):
    for x, val in enumerate(row):
        predicted[y][x] = m(x,y)

rme = np.mean(np.sqrt((predicted - inp)**2))
print('Root mean error:', rme)

show_3d_data(inp)
show_3d_data(predicted)
