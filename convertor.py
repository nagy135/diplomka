#Demo
import numpy as np
from astropy.io import fits
from astropy.utils.data import download_file
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import scipy.stats as st

class FitsConvertor(object):
    def __init__(self, image=None):
        self.image = image

    def show_image(self,image=self.image cmap=plt.get_cmap('gray')):
        plt.imshow(image, cmap=cmap)
        plt.show()

    def load_image(self, image_name):
        self.image = fits.open(image_name)
        self.image = self.image[0].data
        return self.image

    def gauss_kernel(self, kernlen=3, nsig=3):
        interval = (2*nsig+1.)/(kernlen)
        x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw/kernel_raw.sum()
        return kernel

    def convolve(self, size, kernel):
        image = self.image
        if kernel == 'gaussian':
            if size == 3:
                kernel = np.array([[1/16, 1/8, 1/16],[1/8, 1/4, 1/8],[1/16, 1/8, 1/16]])
            if size == 21:
                kernel = self.gauss_kernel(size)
        self.image = convolve2d(image, kernel, mode='valid')




if __name__ == "__main__":
    instance = FitsConvertor()
    #instance.load_image("AGO_2017_PR25_R-005.fit")
    image_file = download_file('http://data.astropy.org/tutorials/FITS-images/HorseHead.fits', cache=True )
    instance.load_image(image_file)
    instance.convolve(21,'gaussian')
    instance.show_image()
