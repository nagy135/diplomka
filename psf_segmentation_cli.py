import argparse
import os
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


from fits_control import read_fits_file, read_fits_file_headers, edit_fits_data
from plotting import show_3d_data, show_data
from PointCluster import PointCluster
from hist_threshold import histogram_threshold

from background import sigma_clipper

from utils import neighbor_check, progressBar

from decorators import print_function, time_function


def threshold_extract_clusters(image, sigma_threshold=2, show_segmentation=False, show_fit_threshold=False):
    threshold = int( histogram_threshold( image, show_fit_threshold, sigma_threshold ))
    noise_dispersion = int( histogram_threshold( image, sigma_only=True ))
    print('threshold level: ', threshold)

    # image = histogram_threshold(image, threshold_sigma=2)

    mask = image > threshold
    image = image.astype('int16')
    image = image - threshold
    image[image<0] = 0
    image = image.astype('uint16')
    if show_segmentation:
        show_data([image,mask], ['image','maska'])
    points = list()

    joined_points = join_neigbor_points_mask(mask)

    print('{} point clusters detected'.format(len(joined_points)))
    clusters = []
    for point_mesh in joined_points:
        cluster = PointCluster(point_mesh, image)
        cluster.noise_dispersion = noise_dispersion
        clusters.append(cluster)
    return clusters

def sobel_extract_clusters(image, show_segmentation=False, threshold=20):
    image_int32 = image.astype('int32')
    dx = ndimage.sobel(image_int32, 0)
    dy = ndimage.sobel(image_int32, 1)
    mag = np.hypot(dx, dy)
    mag *= 255.0 / np.max(mag)
    original_mask = mag
    mask = original_mask >= threshold

    if show_segmentation:
        show_data([image,mask], ['image','maska'])
    points = list()

    joined_points = join_neigbor_points_mask(mask)

    print('{} point clusters detected'.format(len(joined_points)))

    noise_dispersion = int( histogram_threshold( image, sigma_only=True ))

    clusters = []
    for point_mesh in joined_points:
        cluster = PointCluster(point_mesh, image)
        cluster.noise_dispersion = noise_dispersion
        cluster.sobel = True
        clusters.append(cluster)
    return clusters

@time_function('joining of points')
def join_neigbor_points_mask(mask):
    joined_points = list()
    for y, row in enumerate(mask):
        for x, value in enumerate(row):
            if value == 1:
                mask[y][x] = 0
                cluster = list()
                cluster.append((x,y))
                stack = [(x,y)]
                while len(stack) > 0:
                    x_p,y_p = stack.pop()
                    if x_p-1 >= 0 and mask[y_p][x_p-1] == 1:
                        mask[y_p][x_p-1] = 0
                        stack.append((x_p-1,y_p))
                        cluster.append((x_p-1,y_p))
                    if x_p+1 < mask.shape[1] and mask[y_p][x_p+1] == 1:
                        mask[y_p][x_p+1] = 0
                        stack.append((x_p+1,y_p))
                        cluster.append((x_p+1,y_p))
                    if y_p-1 >= 0 and mask[y_p-1][x_p] == 1:
                        mask[y_p-1][x_p] = 0
                        stack.append((x_p,y_p-1))
                        cluster.append((x_p,y_p-1))
                    if y_p+1 < mask.shape[0] and mask[y_p+1][x_p] == 1:
                        mask[y_p+1][x_p] = 0
                        stack.append((x_p,y_p+1))
                        cluster.append((x_p,y_p+1))
                joined_points.append(cluster)
    return joined_points

def join_neigbor_points(points):
    print('joining points into clusters...')
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




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="target of psf extraction")
    parser.add_argument("-f", help="function used for object fitting")
    parser.add_argument("-o", help="file to output data into")
    parser.add_argument("-s", help="segmentation method used")
    parser.add_argument("-sq", help="width,height of square we fit around the maximum of cluster", nargs="+", type=int)
    parser.add_argument("--sigma_threshold", help="number of sigma removed to the right from center of threholding fit onto histogram", type=float)
    parser.add_argument("--sobel_threshold", help="thresholding value for sobel operator segmentation", type=int)
    parser.add_argument("--background_iterations", help="number of iterations during sigma clipping", type=int)
    parser.add_argument("--show_segmentation", help="show the result of segmentation", action="store_true")
    parser.add_argument("--show_object_fit", help="show the result of each object fitting", action="store_true")
    parser.add_argument("--show_object_fit_separate", help="show the result of each object fitting, but firstly original data, then fit", action="store_true")
    parser.add_argument("--show_fit_threshold", help="show the result of thresholding fit", action="store_true")
    parser.add_argument("--show_3d", help="show image in 3d before segmentation", action="store_true")
    args = parser.parse_args()


    show_segmentation = False
    if args.show_segmentation:
        show_segmentation = True

    show_fit_threshold = False
    if args.show_fit_threshold:
        show_fit_threshold = True

    show_object_fit = False
    if args.show_object_fit:
        show_object_fit = True

    show_object_fit_separate = False
    if args.show_object_fit_separate:
        show_object_fit_separate = True

    show_3d = False
    if args.show_3d:
        show_3d = True

    if args.file is None or not os.path.isfile(args.file):
        raise FileNotFoundError('target file not given')
    if args.o is None:
        raise FileNotFoundError('output file not given')
    image = read_fits_file(args.file)
    # image = np.flip(image, axis=0)
    headers  = read_fits_file_headers(args.file)

    if show_3d:
        show_3d_data(image)

    segmentation_options = ['fit_threshold', 'sobel']
    extracted_point_clusters = None
    if not args.s:
        raise AttributeError('need segmentation method, -s')
    if args.s in segmentation_options:
        if args.s == 'fit_threshold':
            if not args.sigma_threshold:
                sigma_threshold = 2
            else:
                sigma_threshold = args.sigma_threshold
            extracted_point_clusters = threshold_extract_clusters( image, show_segmentation=show_segmentation, sigma_threshold=sigma_threshold, show_fit_threshold=show_fit_threshold )
        if args.s == 'sobel':
            if not args.sobel_threshold:
                sobel_threshold = 20
            else:
                sobel_threshold = args.sobel_threshold
            extracted_point_clusters = sobel_extract_clusters(image, show_segmentation=show_segmentation, threshold=sobel_threshold)
    else:
        raise AttributeError('unknown segmentation method, available:', str(segmentation_options))

    fit_options = ['gauss', 'veres']
    fit_function = None
    if not args.f:
        raise AttributeError('need fitting function, -f')
    if args.f in fit_options:
        fit_function = args.f
    else:
        raise AttributeError('unknown fitting function, available:' + str(fit_options))

    number_of_iterations = 2
    if args.background_iterations:
        number_of_iterations = args.background_iterations

    square_size = (11,11)
    if args.sq:
        square_size = args.sq

    print('Calculating image background...')
    background = sigma_clipper(image, number_of_iterations=number_of_iterations)


    output_data = []
    print('Fitting functions to clusters...')
    for i,cluster in enumerate(extracted_point_clusters):
        cluster.show_object_fit = show_object_fit
        cluster.show_object_fit_separate = show_object_fit_separate
        cluster.add_header_data( headers )
        cluster.add_background_data( background )
        try:
            params = cluster.fit_curve(function=fit_function, square_size=square_size)
        except Exception as e:
            continue # suppress all Exceptions, incorrect fits are discarded
        finally:
            if not show_object_fit and not show_object_fit_separate:
                progressBar(i, len(extracted_point_clusters) - 1)

        if cluster.correct_fit:
            output_data.append(cluster.output_data())

    result = ""
    result += '-' * 150 + '\n'
    result += '{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}'.format("x", "y", "flux", "fwhm_x|fwhm_y", "peak_SNR", "fit_rms", "skew_x|skew_y", "kurt_x|kurt_y") + '\n'
    result += '-' * 150 + '\n'
    for i, data in enumerate(output_data):
            result += '{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}'.format(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]) + '\n'

    with open(args.o, 'w') as file:
        file.write(result)
