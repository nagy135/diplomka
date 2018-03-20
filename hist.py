import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt

img = fits.open('AGO_2017_PR25_R-005.fit')
flattened = img[0].data.flatten()

# print(flattened.shape)
# print(flattened[:5])
# plt.xlim([np.min(flattened)-10, np.max(flattened)+10])
x1,x2,y1,y2 = plt.axis()
plt.hist(flattened, np.unique(flattened).shape[0])
plt.axis((np.min(flattened)-10, np.max(flattened)+10, 0, 10000))
plt.show()
