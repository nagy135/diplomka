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
            for y, row in enumerate(mask):
                for x, point in enumerate(row):
                    if point == 1:
                        points_around = recurrent_point_search(x, y, list(), mask)
                        point_spread_mashes.append(PointSpreadMesh(points_around))
                        raise continue_while_loop
        except ContinueWhileLoop:
            continue

def recurrent_point_search(x, y, points, mask):
    if mask[y][x] == 1 and (x,y) not in points:
        points.append((x,y))
    new_points = list()
    if x-1 >= 0:
        new_points.append(recurrent_point_search(x-1, y, points, mask))
    if x+1 < mask.shape[1]:
        new_points.append(recurrent_point_search(x+1, y, points, mask))
    if y-1 >= 0:
        new_points.append(recurrent_point_search(x, y-1, points, mask))
    if x+1 < mask.shape[0]:
        new_points.append(recurrent_point_search(x, y+1, points, mask))


    return points + new_points


image = read_fits_file('M27_R_60s-001.fit')
# create_psf_objects(image)
show_3d_data(image)
