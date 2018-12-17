import numpy as np
import copy
import cv2
from astropy.io import fits
from astropy.utils.data import download_file
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import scipy.stats as st

def pipeline( img_src ):
    data_and_headers = fits.open('data/' + img_src)
    data = data_and_headers[0].data
    print('Astonomical image with shape {} loaded'.format(data.shape))

    data = convolve(data, 21, 'gaussian')
    # data = convolve(data, 3, 'gaussian')

    data = extract_lines(data)

    return data

def show_histogram( image ):
    flattened = image.flatten()
    x1,x2,y1,y2 = plt.axis()
    plt.hist(flattened, np.unique(flattened).shape[0])
    plt.axis((np.min(flattened)-10, np.max(flattened)+10, 0, 10000))
    plt.show()

def show_image( image, name):
    print('Showing img')
    plt.imshow(image, cmap='gray')
    plt.title(name), plt.xticks([]), plt.yticks([])
    plt.show()

def gauss_kernel( kernlen=3, nsig=3):
    print('Creating kernel with size : {}'.format(str(kernlen)))
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

def psf(data):
    print('Applying point spread function (PSF)')
    pass

def convolve( image, size, kernel_recipe='gaussian'):
    kernel = None
    if kernel_recipe == 'gaussian':
        if size == 3:
            kernel = np.array([[1/16, 1/8, 1/16],[1/8, 1/4, 1/8],[1/16, 1/8, 1/16]])
        else:
            kernel = gauss_kernel(size)
    if kernel is None:
        raise Exception('Unknown kernel')
    print('Applying 2d convolution')
    image = convolve2d(image, kernel, mode='same', boundary='fill')
    return image

def extract_lines( image ):
    print('Extracting lines')
    image_copy = copy.deepcopy(image)
    image = np.uint8(image)
    if image.ndim > 2:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(image,50,150,apertureSize = 7)
    minLineLength = 5
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges,1,np.pi/180,100 ,minLineLength,maxLineGap).reshape((-1,4))
    print('{} lines detected via Hough'.format(len(lines)))
    black_n_lines = np.zeros_like(image_copy)
    for x1,y1,x2,y2 in lines:
        cv2.line(black_n_lines,(x1,y1),(x2,y2),(255, 255, 255),2)
    return black_n_lines



result = pipeline('sample.fit')
# show_histogram(result)
show_image(result, 'Final Image')
