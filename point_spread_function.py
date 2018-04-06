import math
import random
import numpy as np
import copy
from astropy.io import fits
from astropy.utils.data import download_file
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.misc import imresize
import scipy.stats as st
from scipy.signal import medfilt2d


from fits_control import read_fits_file, edit_fits_data, show_image
from plot3d import show_3d_data
from PointSpreadMesh import PointSpreadMesh

from decorators import print_function, time_function

@print_function('Starting creation of PSF objects')
def create_psf_objects(image):
    threshold = np.mean(np.array([np.max(image), np.median(image)])) * 1/2

    mask = image > threshold
    # show_image([image,mask], ['image','maska'])
    points = list()

    for y, row in enumerate(mask):
        for x, point in enumerate(row):
            if point == 1:
                points.append((x,y))
    print('Algorithm detected {} points'.format(str(len(points))))

def neighbor_check(first_point, second_point):
    dist = np.linalg.norm( np.array(first_point) - np.array(second_point) )
    if dist == 1 or dist == math.sqrt(2):
        return True
    return False


image = read_fits_file('M27_R_60s-001.fit')
# image = np.array([
#     [0,0,0,0],
#     [0,1000,1000,0],
#     [0,1000,1000,0],
#     [0,0,0,0],
#     ])
# create_psf_objects(image)
# show_3d_data(image)
