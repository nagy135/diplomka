import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt

img = fits.open('data/M27_R_60s-002.fit')
f = np.fft.fft2(img[0].data)
print(img[0].data.shape)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

plt.subplot(121),plt.imshow(img[0].data, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
