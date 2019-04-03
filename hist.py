import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt

img = fits.open('data/M27_R_60s-002.fit')
flattened = img[0].data.flatten().astype(int)

# print(np.max(flattened))
# print(np.min(flattened))
flattened += abs(np.min(flattened))
# assert False ,'die'
# print(flattened[:5])
plt.xlim([np.min(flattened)-10, np.max(flattened)+10])
# plt.ylim([0, 20])
x1,x2,y1,y2 = plt.axis()
n, bins, patches = plt.hist(flattened, np.unique(flattened).shape[0])
plt.axis((np.min(flattened)-10, np.max(flattened)+10, 0, 10000))
print(bins)
plt.show()
