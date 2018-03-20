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

from plot3d import show_3d_data



def create_artificial_background(image):
    background = np.zeros(image.shape)
    width = image.shape[0] //2
    height = image.shape[1] //2
    background[height:,width:] = -200
    background[height-100:height+100,width-100:width+100] = -200
    background = convolve(background, 86, 'gaussian')
    np.save('data/background_map.npy', background)
    return background

def read_fits_file(fits_file):
    data_and_headers = fits.open('data/' + fits_file)
    data = data_and_headers[0].data
    print('Astonomical image with shape {} loaded'.format(data.shape))
    data_and_headers.close()
    return data

def edit_fits_data(fits_file, new_data, new_file_name):
    data_and_headers = fits.open('data/' + fits_file)
    data_and_headers[0].data = new_data
    data_and_headers[0].writeto('data/' + new_file_name)
    return

def convolve( image, size, kernel_recipe='gaussian'):
    kernel = None
    if kernel_recipe == 'gaussian':
        if size == 3:
            kernel = np.array([[1/16, 1/8, 1/16],[1/8, 1/4, 1/8],[1/16, 1/8, 1/16]])
        else:
            kernel = gauss_kernel(size)
    if kernel is None:
        raise Exception('Unknown kernel')
    print('Applying 2d convolution')
    image = convolve2d(image, kernel, mode='same', boundary='fill')
    return image

def gauss_kernel( kernlen=3, nsig=3):
    print('Creating kernel with size : {}'.format(str(kernlen)))
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

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

def iterative_sigma_clipping(original_image, preprocessed_image, curr_iter, last_iter_background):
    print('==================================================')
    print('Performing iterative sigma clipping (iteration {})'.format(str(curr_iter)))
    if curr_iter == 0:
        return original_image - preprocessed_image
    else:
        standard_deviation = np.std(last_iter_background)
        mean_deviation = np.median(last_iter_background)
        print('previous mean was {}'.format(str(mean_deviation)))
        print('previous sigma was {}'.format(str(standard_deviation)))

        new_iter_background = np.zeros(last_iter_background.shape)
        for num_col in range(last_iter_background.shape[0]):
            for num_row in range(last_iter_background.shape[1]):
                term = np.absolute(last_iter_background[num_col,num_row] - mean_deviation)
                if term < 3*standard_deviation:
                    new_iter_background[num_col,num_row] = last_iter_background[num_col,num_row]
                else:
                    new_iter_background[num_col,num_row] = mean_deviation
        return new_iter_background



def perform_sigma_clipping(original_image, number_of_iterations=9):
    preprocessed_image = image_preprocess(original_image)
    assert original_image.shape == preprocessed_image.shape

    estimated_background = np.zeros(original_image.shape)
    last_iter_background = None
    for curr_iter in range(number_of_iterations):
        estimated_background = iterative_sigma_clipping(original_image, preprocessed_image, curr_iter, last_iter_background)
        print('mean from this iteration : {}'.format(str(np.mean(estimated_background))))
        last_iter_background = estimated_background
    print(estimated_background)
    print(preprocessed_image)
    return preprocessed_image + estimated_background



def image_preprocess(image):
    initial_shape = image.shape
    # image = np.full(image.shape, 500)
    image = imresize(image, 0.1, interp='bicubic')
    image = imresize(image, initial_shape,interp='bicubic')
    image = medfilt2d(image, 3)
    image = convolve(image, 3)
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

def sigma_clipper( image, num_tiles_width = 1, num_tiles_height = 1 ):
    if num_tiles_width != 1 or num_tiles_height != 1:
        tile_rows = np.array_split(image, num_tiles_height)
        final = np.zeros(image.shape)
        curr_x = 0
        curr_y = 0
        for row_i,row in enumerate(tile_rows):
            tiles_in_row = np.array_split(row, num_tiles_width, axis=1)
            tile_shape_1 = None
            for col_i, tile in enumerate(tiles_in_row):
                tile = perform_sigma_clipping(tile)
                final[curr_y:(curr_y+tile.shape[0]),curr_x:(curr_x+tile.shape[1])] = tile
                curr_x += tile.shape[1]
                tile_shape_1 = tile.shape[0]
            curr_y += tile_shape_1
            curr_x = 0
    else:
        final = perform_sigma_clipping(image)
    return final

image = read_fits_file('STREAK_test_1-023.fit')
print('Mean before : {}'.format(np.mean(image)))
print('Max before : {}'.format(np.max(image)))
print('Min before : {}'.format(np.min(image)))
print('Median before : {}'.format(np.median(image)))
print(image)

# show_hist(image)

# image_contaminated = create_artificial_background(image)
# background_map = np.load('data/background_map.npy')
# copy_image = image
# image = image+image_contaminated

# image = np.arange(100).reshape((10,10))

# extracted_background = sigma_clipper(image, 10, 10)
# extracted_background = sigma_clipper(image)
extracted_background = sigma_clipper(image)
extracted_background = extracted_background.astype('uint16')
result = image-extracted_background
# print(np.max(image))
# print(np.min(image))
# print(np.max(extracted_background))
# print(np.min(extracted_background))
# print(np.max(result))
# print(np.min(result))
# print(image.dtype)
# print(extracted_background.dtype)
# print(result.dtype)
# show_image(result, 'extracted')
# edit_fits_data('M27_R_60s-001.fit', image, 'original_image.fit')
# edit_fits_data('M27_R_60s-001.fit', extracted_background, 'extracted_bg.fit')
edit_fits_data('STREAK_test_1-023.fit', extracted_background, 'linear_bg_3_edited_median.fit')
edit_fits_data('STREAK_test_1-023.fit', result, 'linear_result_3_edited_median.fit')
# minimum = 1600
# maximum = 3000
# result[minimum>result] = minimum
# result[maximum<result] = maximum
# image[minimum>image] = minimum
# image[maximum<image] = maximum
# show_image([image,extracted_background,result],['image','bg', 'result'])
# show_image([image,extracted_background, result], ['before','after', 'extracted'])
# show_image(image-extracted_background, 'after extraction')
