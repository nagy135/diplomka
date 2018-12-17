import math
import random
import numpy as np
import copy
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.optimize import curve_fit

from fits_control import read_fits_file, edit_fits_data, show_image

from decorators import print_function, time_function

def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x-cen)**2 / wid)

def histogram_threshold(image, show=False, threshold_sigma=2):
    # image = image / 1000
    hist, bins = np.histogram(image.flatten(), bins=len(np.unique(image)), density=True)

    init_vals = [1000., 0., 1000.]

    best_vals, covar = curve_fit(gaussian, [x for x in range(len(hist))], hist, p0=init_vals)

    # Get the fitted curve
    hist_fit = gaussian([x for x in range(len(hist))], *best_vals)

    center = best_vals[1]
    offset = best_vals[2] * threshold_sigma
    threshold = center + offset
    if show:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # smoothed = smooth(hist,15)
        # ax.bar([x for x in range(len(hist))], hist)
        ax.axvline(center + offset, color='green')
        # ax.plot([x for x in range(len(hist))], hist, color="green")
        ax.plot([x for x in range(len(hist))], hist, label='Test data', color='red')
        ax.plot([x for x in range(len(hist))], hist_fit, label='Fitted data', color='blue')
        plt.show()
    else:
        image[image < offset] = 0
        return image



    # show_image(image, 'image')



if __name__ == '__main__':
    # image = read_fits_file('data/AGO_2017_PR25_R-005.fit')
    image = read_fits_file('data/STREAK_test_1-002.fit')
    histogram_threshold(image, True)