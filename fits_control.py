import random
import numpy as np
import copy
import cv2
import os.path
from astropy.io import fits
from astropy.utils.data import download_file
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.misc import imresize
import scipy.stats as st
from scipy.signal import medfilt2d

def read_fits_file(fits_file, subfolder=''):
    data_and_headers = fits.open(subfolder + fits_file)
    data = data_and_headers[0].data
    print('Astonomical image with shape {} loaded'.format(data.shape))
    data_and_headers.close()
    return data

def edit_fits_data(fits_file, new_data, new_file_name, subfolder="", input_subfolder="", comment=""):
    if os.path.isfile(os.getcwd() + '/' + new_file_name):
        os.remove(os.getcwd() + '/' + new_file_name)
    data_and_headers = fits.open(input_subfolder + fits_file)
    data_and_headers[0].data = new_data
    if comment != "":
        header = data_and_headers[0].header
        header['Comment'] = comment
    data_and_headers[0].writeto(os.getcwd() + '/' + new_file_name)
    return

def show_image( image, name):
    print('Showing img')
    if type(image) is type(list()):
        fig, ax = plt.subplots(nrows=1, ncols=len(image))
        for i,row in enumerate(ax):
            row.imshow(image[i], cmap='gray')
            # row.title(name[i]), plt.xticks([]), plt.yticks([])
        plt.show()
    else:
        plt.imshow(image, cmap='gray')
        plt.title(name), plt.xticks([]), plt.yticks([])
        plt.show()
