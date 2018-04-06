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

class ContinueWhileLoop(Exception):
    pass
continue_while_loop = ContinueWhileLoop()

@print_function('Starting creation of PSF objects')
def create_psf_objects(image):
    threshold = np.mean(np.array([np.max(image), np.median(image)])) * 1/2

    mask = image > threshold
    # show_image([image,mask], ['image','maska'])
    point_spread_mashes = list()

    mask_sum = mask.sum()
    while mask_sum > 0:
        try:
            print('Points remaining in mask : {}'.format(str(mask_sum)))
            for y, row in enumerate(mask):
                for x, point in enumerate(row):
                    if point == 1:
                        points_around = recurrent_point_search(x, y, [], mask)
                        mask_sum -= len(points_around)
                        point_spread_mashes.append(PointSpreadMesh(points_around))
                        raise continue_while_loop
        except ContinueWhileLoop:
            continue
    print('Algorithm detected {} point meshes'.format(str(len(point_spread_mashes))))

def recurrent_point_search(x, y, points, mask):
    if mask[y][x] == 1 and (x,y) not in points:
        points.append((x,y))
        mask[y][x] = 0
    else:
        return []
    new_points = []
    if x-1 >= 0:
        new_points += recurrent_point_search(x-1, y, points, mask)
    if x+1 < mask.shape[1]:
        new_points += recurrent_point_search(x+1, y, points, mask)
    if y-1 >= 0:
        new_points += recurrent_point_search(x, y-1, points, mask)
    if y+1 < mask.shape[0]:
        new_points += recurrent_point_search(x, y+1, points, mask)

    return list(set(points + new_points))


# image = read_fits_file('M27_R_60s-001.fit')
image = np.array([
    [0,0,0,0],
    [0,1000,1000,0],
    [0,1000,1000,0],
    [0,0,0,0],
    ])
create_psf_objects(image)
# show_3d_data(image)
