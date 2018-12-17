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

from hist_threshold import histogram_threshold

from decorators import print_function, time_function

@print_function('Starting creation of PSF objects')
def extract_point_spread_meshes(image):
    threshold = np.mean(np.array([np.max(image), np.median(image)])) * 1/2

    image = histogram_threshold(image, threshold_sigma=2)

    mask = image > threshold
    show_image([image,mask], ['image','maska'])
    points = list()

    for y, row in enumerate(mask):
        for x, point in enumerate(row):
            if point == 1:
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
image = read_fits_file('data/AGO_2017_PR25_R-005.fit')
extracted_point_spread_meshes= []
extract_point_spread_meshes(image)
for point_mash in extracted_point_spread_meshes:
    params = point_mash.fit_curve()
