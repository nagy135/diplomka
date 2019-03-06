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
from scipy import ndimage


from fits_control import read_fits_file, edit_fits_data, show_image
from plot3d import show_3d_data
from PointSpreadMesh import PointSpreadMesh

from hist_threshold import histogram_threshold

from decorators import print_function, time_function

def extract_point_spread_meshes(image):
    threshold = 50
    image_int32 = image.astype('int32')
    dx = ndimage.sobel(image_int32, 0)
    dy = ndimage.sobel(image_int32, 1)
    mag = np.hypot(dx, dy)
    mag *= 255.0 / np.max(mag)
    original_mask = mag
    mask = original_mask >= threshold

    show_image([original_mask,mask], ['image','maska'])
    points = list()

    for y, row in enumerate(mask):
        for x, point in enumerate(row):
            if point:
                points.append((x,y))
    print('Algorithm detected {} points'.format(str(len(points))))
    joined_points = join_neigbor_points(points)
    print('{} point meshes detected'.format(len(joined_points)))
    point_meshes = []
    for point_mesh in joined_points:
        point_meshes.append(PointSpreadMesh(point_mesh, image))
    global extracted_point_spread_meshes
    extracted_point_spread_meshes = point_meshes


def join_neigbor_points(points):
    joined = list()
    extracted_point = points.pop()
    joined.append([extracted_point])
    while len(points) > 0:
        point = points.pop()
        found_spot = False
        for i,point_mesh in enumerate(joined):
            if found_spot:
                break
            for u,point_mesh_point in enumerate(point_mesh):
                if found_spot:
                    break
                if neighbor_check(point_mesh_point, point):
                    found_spot = True
                    joined[i].append(point)
        if not found_spot:
            joined.append([point])
    return joined


def neighbor_check(first_point, second_point):
    dist = np.linalg.norm( np.array(first_point) - np.array(second_point) )
    if dist == 1 or dist == math.sqrt(2):
        return True
    return False


# image = read_fits_file('data/M27_R_60s-001.fit')
image = read_fits_file('data/M27_R_60s-002.fit')
# image = read_fits_file('data/STREAK_test_1-003.fit')
extracted_point_spread_meshes= []
extract_point_spread_meshes(image)
for point_mash in extracted_point_spread_meshes:
    params = point_mash.fit_curve()
    show_image(point_mash.squared_data, 'point mash')
