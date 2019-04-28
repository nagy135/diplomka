import random
import numpy as np
import copy
import cv2
import os
from astropy.io import fits
from astropy.utils.data import download_file
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.misc import imresize
import scipy.stats as st
from scipy.signal import medfilt2d

from fits_control import read_fits_file, edit_fits_data

from plotting import show_3d_data

def create_artificial_background(image):
    background = np.zeros(image.shape)
    width = image.shape[0] //2
    height = image.shape[1] //2
    background[height:,width:] = -200
    background[height-100:height+100,width-100:width+100] = -200
    background = convolve(background, 86, 'gaussian')
    np.save('data/background_map.npy', background)
    return background

def convolve( image, size, kernel_recipe='gaussian'):
    kernel = None
    if kernel_recipe == 'gaussian':
        if size == 3:
            kernel = np.array([[1/16, 1/8, 1/16],[1/8, 1/4, 1/8],[1/16, 1/8, 1/16]])
        else:
            kernel = gauss_kernel(size)
    if kernel is None:
        raise Exception('Unknown kernel')
    image = convolve2d(image, kernel, mode='same', boundary='fill')
    return image

def gauss_kernel( kernlen=3, nsig=3):
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

def iterative_sigma_clipping(original_image, preprocessed_image, curr_iter, last_iter_background):
    if curr_iter == 0:
        return original_image - preprocessed_image
    else:
        standard_deviation = np.std(last_iter_background)
        mean_deviation = np.median(last_iter_background)

        new_iter_background = np.zeros(last_iter_background.shape)
        for num_col in range(last_iter_background.shape[0]):
            for num_row in range(last_iter_background.shape[1]):
                term = np.absolute(last_iter_background[num_col,num_row] - mean_deviation)
                if term < 3*standard_deviation:
                    new_iter_background[num_col,num_row] = last_iter_background[num_col,num_row]
                else:
                    new_iter_background[num_col,num_row] = mean_deviation
        return new_iter_background



def perform_sigma_clipping(original_image, number_of_iterations=2):
    preprocessed_image = image_preprocess(original_image)
    assert original_image.shape == preprocessed_image.shape

    estimated_background = np.zeros(original_image.shape)
    last_iter_background = None
    for curr_iter in range(number_of_iterations):
        estimated_background = iterative_sigma_clipping(original_image, preprocessed_image, curr_iter, last_iter_background)
        last_iter_background = estimated_background
    return preprocessed_image + estimated_background



def image_preprocess(image):
    initial_shape = image.shape
    # image = np.full(image.shape, 500)
    image = imresize(image, 0.1, interp='bicubic')
    image = medfilt2d(image, 15)
    image = imresize(image, initial_shape,interp='bicubic')
    image = convolve(image, 15)
    # image = image + random.randint(5,500)
    return image

def show_hist(image):
    flattened = image.flatten()
    print('max value is {}'.format(np.max(image)))
    print('type is {}'.format(image.dtype))
    x1,x2,y1,y2 = plt.axis()
    plt.hist(flattened, np.unique(flattened).shape[0])
    plt.axis((np.min(flattened)-10, np.max(flattened)+10, 0, 10000))
    plt.show()

def sigma_clipper( image, num_tiles_width = 1, num_tiles_height = 1, number_of_iterations=2 ):
    if num_tiles_width != 1 or num_tiles_height != 1:
        tile_rows = np.array_split(image, num_tiles_height)
        final = np.zeros(image.shape)
        curr_x = 0
        curr_y = 0
        for row_i,row in enumerate(tile_rows):
            tiles_in_row = np.array_split(row, num_tiles_width, axis=1)
            tile_shape_1 = None
            for col_i, tile in enumerate(tiles_in_row):
                tile = perform_sigma_clipping(tile, number_of_iterations=number_of_iterations)
                final[curr_y:(curr_y+tile.shape[0]),curr_x:(curr_x+tile.shape[1])] = tile
                curr_x += tile.shape[1]
                tile_shape_1 = tile.shape[0]
            curr_y += tile_shape_1
            curr_x = 0
    else:
        final = perform_sigma_clipping(image, number_of_iterations=number_of_iterations)
    return cv2.blur(final,(35,35))

def fix_sizes(a1, a2):
    if a1.shape == a2.shape:
        return a1,a2
    if a1.shape[0] > a2.shape[0] and a1.shape[1] > a2.shape[1]:
        bigger = a1
        smaller = a2
    else:
        bigger = a2
        smaller = a1
    difference = (bigger.shape[0] - smaller.shape[0],bigger.shape[1] - smaller.shape[1])
    first = difference[0] // 2
    second = difference[1] // 2
    print('Fixing sizes of outputs with shapes {} and {}'.format(str(bigger.shape), str(smaller.shape)))
    print('Crop {} from edges in 1 dimension and {} in second'.format(str(first), str(second)))
    if bigger is a1:
        return a1[first:-first, second:-second], a2
    else:
        return a1, a2[first:-first, second:-second]

if __name__ == '__main__':
    for i in range(8,9):
        input_file = 'data/generated/Comb_' + str(i) + '/Comb/Comb_' + str(i) + '.fits'
        image = read_fits_file(input_file)

        original_background_input_file = 'data/generated/Comb_' + str(i) + '/Noise/Noise_' + str(i) + '.fits'
        original_background = read_fits_file(original_background_input_file)
        # show_3d_data(original_background, method='matplotlib')

        file_name, extension = os.path.basename(input_file).split('.')
        # extracted_background = sigma_clipper(image, 50, 50)
        extracted_background = sigma_clipper(image)

        difference_between_backgrounds = abs(original_background - extracted_background)

        edit_fits_data(input_file, extracted_background, file_name+'_35x35_bg.'+extension, 'MultiExport/')

        edit_fits_data(input_file, difference_between_backgrounds, file_name+'_35x35_difference_bgs.'+extension, 'MultiExport/')
        # extracted_background = sigma_clipper(extracted_background)
        # show_3d_data(extracted_background, method='matplotlib')
        # edit_fits_data(input_file, extracted_background, file_name+'bg2.'+extension)

        # extracted_background = extracted_background.astype('uint32')

        # image, extracted_background = fix_sizes(image, extracted_background)
        result = image-extracted_background

        edit_fits_data(input_file, result, file_name+'_35x35_result.'+extension, 'MultiExport/')

        show_3d_data(image, method='matplotlib')
        show_3d_data(extracted_background, method='matplotlib')
        show_3d_data(result, method='matplotlib')

        # file_name, extension = os.path.basename(input_file).split('.')
        # edit_fits_data(input_file, extracted_background, file_name+'bg.'+extension)
        # edit_fits_data(input_file, result, file_name+'result.'+extension)
