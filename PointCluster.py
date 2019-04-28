import numpy as np
import math
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
from scipy import optimize
from scipy.special import erf, seterr
from scipy.stats import kurtosis, skew
from astropy.modeling import models, fitting
import scipy.integrate as integrate
import warnings

from utils import normalize, rms, psnr
from plotting import *

seterr(all='ignore') # suppress fitting errors
warnings.simplefilter("ignore") # suppress fitting warnings

class PointCluster(object):
    ''' Object representing single cluster of pixels, fits functions as well as produces output for object '''

    def __init__(self, points, image):
        self.points = points
        self.correct_fit = False
        self.peak_point = None
        self.header_data = None
        self.background_data = None
        self.squared_data = None
        self.image = image
        self.kurtosis = None
        self.skew = None
        self.psnr = None
        self.show_object_fit = False
        self.sobel = False
        self.noise_median = 0

    def __repr__(self):
        '''x y Flux  FWHM PeakSNR RMS Skew Kurtosis'''
        return str([self.x0, self.y0, self.cumulated_flux, self.fwhm, self.psnr, self.rms, self.skew, self.kurtosis])

    def output_data(self):
        '''x y Flux  FWHM PeakSNR RMS Skew Kurtosis'''
        return [abs(self.x0), abs(self.y0), self.cumulated_flux, self.fwhm, self.psnr, self.rms, self.skew, self.kurtosis]

    def add_header_data( self, header_data ):
        self.header_data = header_data

    def add_background_data( self, background_data ):
        self.background_data_raw = background_data

    def gaussian_2d(self, data_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
        (x, y) = data_tuple
        xo = float(xo)
        yo = float(yo)
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
        return g.ravel()

    def veres(self, data_tuple, x0, y0, width, length, rotation, total_flux):
        # (x, y) = data_tuple
        x0 = float(x0)
        y0 = float(y0)

        res_arr = np.zeros_like(data_tuple[0])
        print('params are: ', x0, y0,'len:', length,'width:', width, 'total_flux:', total_flux)

        for y_init in range(data_tuple[0].shape[0]):
            for x_init in range(data_tuple[0].shape[1]):
                x = (x_init-x0) * math.cos(math.radians(rotation)) - (y_init-y0)*math.sin(math.radians(rotation))
                y = (x_init-x0) * math.sin(math.radians(rotation)) + (y_init-y0)*math.cos(math.radians(rotation))
                first_term = total_flux / length
                second_term = (1/(math.sqrt(2*math.pi*width**2)))
                third_term = integrate.quad(lambda l: np.exp(-1/(2*width**2)*((x-l)**2+y**2)) ,-length//2, length//2)

                res = self.background_data[y_init][x_init] + first_term * second_term * third_term[0]
                res_arr[y_init][x_init] = res
        # show_3d_data(res_arr, secondary_data=[self.squared_data])
        return res_arr.ravel()

    def moments(self, data):
        """Returns (height, x, y, width_x, width_y)
        the gaussian parameters of a 2D distribution by calculating its
        moments """
        total = data.sum()
        X, Y = np.indices(data.shape)
        x = (X*data).sum()/total
        y = (Y*data).sum()/total
        col = data[:, int(y)]
        width_x = np.sqrt(np.abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
        row = data[int(x), :]
        width_y = np.sqrt(np.abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
        height = data.max()
        return height, x, y, width_x, width_y

    def fill_to_square(self, square_width, square_height, center=None):
        max_value = 0
        self.low_y, self.low_x = self.image.shape
        for point in self.points:
            if self.image[point[1],point[0]] >= max_value:
                self.peak_point = point
                max_value = self.image[point[1],point[0]]
            if point[1] <= self.low_y:
                self.low_y = point[1]
            if point[0] <= self.low_x:
                self.low_x = point[0]
        if center is not None:
            self.peak_point = (self.peak_point[0] + (center[0] - square_width//2), self.peak_point[1] + (center[1] - square_height//2))
        square = np.zeros((square_height, square_width))
        mid_x, mid_y = square_width // 2, square_height // 2
        for y in range(square_height):
            for x in range(square_width):
                relative_x = self.peak_point[0] - mid_x + x
                relative_y = self.peak_point[1] - mid_y + y
                square[y,x] = self.image[relative_y,relative_x]
        self.background_data = np.zeros((square_height, square_width))
        self.low_x = self.peak_point[0] - (square_width//2)
        self.low_y = self.peak_point[1] - (square_height//2)
        for y, row in enumerate(self.background_data):
            for x, pixel in enumerate(row):
                self.background_data[y][x] = self.background_data_raw[y+self.low_y][x+self.low_x]
        return square


    def fit_curve(self, function='gauss', square_size=(11,11)):
        try:
            self.squared_data = self.fill_to_square(*square_size)
        except IndexError:
            raise IndexError("Border object, ignore")

        if self.sobel:
            self.noise_median = np.median(self.background_data)

        if function == 'gauss':
            x = np.linspace(0, square_size[0]-1, square_size[0])
            y = np.linspace(0, square_size[1]-1, square_size[1])
            x, y = np.meshgrid(x, y)

            if self.squared_data.sum() == 0:
                self.correct_fit = False
                return
            moments = self.moments(self.squared_data)
            pred = [*moments, 10, 0]
            popt, pcov = curve_fit(self.gaussian_2d, (x, y), self.squared_data.flatten(), maxfev=100000, xtol=1e-10, ftol=1e-10, p0=pred)
            try:
                if popt is not None and popt[3] is not None and popt[3] != 0:
                    self.correct_fit = True
                else:
                    self.correct_fit = False
                    return
            except NameError:
                self.correct_fit = False
                return
            self.fwhm_x = 2*math.sqrt(2*math.log(2)) * abs(popt[3])
            self.fwhm_y = 2*math.sqrt(2*math.log(2)) * abs(popt[4])
            self.x0 = round(self.low_x + abs(popt[1]),2)
            self.y0 = round(self.low_y + abs(popt[2]),2)
            if self.x0 >= self.image.shape[1] or \
            self.y0 >= self.image.shape[0] or \
            math.isnan(self.fwhm_x) or \
            math.isnan(self.fwhm_y) or \
            self.fwhm_x >= self.image.shape[1] or \
            self.fwhm_y >= self.image.shape[0]:
                self.correct_fit = False
                return
            self.fwhm = "{}|{}".format(abs(round(self.fwhm_x, 2)),abs(round(self.fwhm_y, 2)))

            self.predicted = self.gaussian_2d((x, y), *popt).reshape(*self.squared_data.shape)
            self.rms_res = rms(self.squared_data, self.predicted)
            if self.show_object_fit or self.show_object_fit_separate:
                print('==============')
                print('Root mean error:', self.rms_res)
                print(self.fwhm)
                if self.show_object_fit_separate:
                    show_3d_data(self.squared_data)
                    show_3d_data(self.predicted, color='red')
                else:
                    show_3d_data(self.squared_data, secondary_data=[self.predicted])

        elif function == 'veres':
            self.length = 50 # from self.header_data
            self.width = 0.5 # from self.header_data
            self.rotation = 45 # rotation from self.header_data
            print('################')
            x = np.linspace(0, square_size[0]-1, square_size[0])
            y = np.linspace(0, square_size[1]-1, square_size[1])
            x, y = np.meshgrid(x, y)

            # gaussian
            moments = self.moments(self.squared_data)
            pred = [*moments, 10, 0]
            popt, pcov = curve_fit(self.gaussian_2d, (x, y), self.squared_data.flatten(), maxfev=500000000, xtol=1e-15, ftol=1e-15, p0=pred)
            predicted_gauss = self.gaussian_2d((x, y), *popt).reshape(*self.squared_data.shape)
            show_3d_data(self.squared_data, secondary_data=[predicted_gauss])
            self.x0 = popt[1]
            self.y0 = popt[2]
            # self.width = min(popt[3],popt[4])
            # gaussian
            if popt[1] < 0 or popt[2] < 0:
                raise IndexError("Incorrect gaussian fit")

            new_center = (int(popt[1]), int(popt[2]))
            try:
                self.squared_data = self.fill_to_square(*square_size, new_center)
            except IndexError:
                raise IndexError("Border object, ignore")

            total_flux = self.squared_data - (np.median(self.background_data)+100)
            total_flux = total_flux[total_flux>0].sum() // 3
            prediction = [square_size[0]//2, square_size[1]//2, 1.5, 55, 45, total_flux]
            # popt, pcov = curve_fit(self.veres, np.array((x, y), dtype=int), self.squared_data.flatten(), maxfev=500000000 )
            veres_bounds = ([0,0,0,0,0,0],[self.squared_data.shape[1], self.squared_data.shape[0], 10, 70, 90, total_flux])
            popt, pcov = curve_fit(self.veres, np.array((x, y), dtype=int), self.squared_data.flatten(), maxfev=500000000, p0=prediction, bounds=veres_bounds )
            # print(pcov)

            # self.predicted = self.veres((x, y), *prediction).reshape(*self.squared_data.shape)
            self.predicted = self.veres((x, y), *popt).reshape(*self.squared_data.shape)
            # self.fwhm = "{}|{}".format(abs(round(popt[2], 2)),abs(round(popt[3], 2)))
            self.fwhm = "unknown"

            self.rms_res = rms(self.squared_data, self.predicted)
            if self.show_object_fit:
                print('==============')
                print('Root mean error:', rms(self.squared_data, self.predicted))
                show_3d_data(self.squared_data, secondary_data=[self.predicted])

        self.cumulated_flux = round(self.squared_data.sum())
        skew_mid_x = round(skew(self.squared_data, 1)[square_size[1]//2], 2)
        skew_mid_y = round(skew(self.squared_data, 0)[square_size[0]//2], 2)
        kurtosis_mid_x = round(kurtosis(self.squared_data, 1, fisher=True)[square_size[1]//2], 2)
        kurtosis_mid_y = round(kurtosis(self.squared_data, 0, fisher=True)[square_size[0]//2], 2)
        self.skew = str(skew_mid_x) + "|" + str(skew_mid_y)
        self.kurtosis = str(kurtosis_mid_x) + "|" + str(kurtosis_mid_y)
        self.rms = round(self.rms_res, 3)
        self.psnr = psnr(self.squared_data, self.noise_median, 5)

if __name__ == '__main__':
    square_size = (50,50)
    x = np.linspace(0, square_size[0]-1, square_size[0])
    y = np.linspace(0, square_size[1]-1, square_size[1])
    x, y = np.meshgrid(x, y)
    veresed = veres((x,y), 25, 25, 40, 5, 200, 0)
    show_3d_data(veresed)
