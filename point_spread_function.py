import random
import numpy as np
import copy
import cv2
from astropy.io import fits
from astropy.utils.data import download_file
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.misc import imresize
import scipy.stats as st
from scipy.signal import medfilt2d


from fits_control import read_fits_file, edit_fits_data, show_image
from PointSpreadMesh import PointSpreadMesh

def create_psf_objects(image):
    a = PointSpreadMesh([1,2,3])
    threshold = np.mean(np.array([np.max(image), np.median(image)])) * 1/3

    mask = image > threshold
    show_image([image,mask], ['image','maska'])
    point_spread_mashes = list()
    while mask.sum() > 0:
        print(mask.sum())
        for i, row in enumerate(mask):
            u, point in enumerate(row):
                pass

image = read_fits_file('M27_R_60s-001.fit')
create_psf_objects(image)
# show_image(image, 'test')
