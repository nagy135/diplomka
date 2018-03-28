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

class ContinueWhileLoop(Exception):
    pass
continue_while_loop = ContinueWhileLoop()

def create_psf_objects(image):
    threshold = np.mean(np.array([np.max(image), np.median(image)])) * 1/3

    mask = image > threshold
    show_image([image,mask], ['image','maska'])
    point_spread_mashes = list()
    while mask.sum() > 0:
        try:
            print(mask.sum())
            for i, row in enumerate(mask):
                for u, point in enumerate(row):
                    if point == 1:
                        points_around = recurent_point_search(i, u, list(), mask)
                        point_spread_mashes.append(PointSpreadMesh(points_around))
                        raise continue_while_loop
        except ContinueWhileLoop:
            continue

def recurent_point_search(i, u, points, mask):
    # TODO
    return recurent_point_search(i-1, u, points, mask)


image = read_fits_file('M27_R_60s-001.fit')
create_psf_objects(image)
# show_3d_data(image, 'test')
