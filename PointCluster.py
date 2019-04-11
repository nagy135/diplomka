import numpy as np
import math
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
from scipy import optimize
from scipy.special import erf
from scipy.stats import kurtosis, skew
from plot3d import *
from astropy.modeling import models, fitting

from utils import normalized, rms



class PointCluster(object):

    def __init__(self, points, image):
        self.points = points
        self.peak_point = None
        self.header_data = None
        self.background_data = None
        self.squared_data = None
        self.image = image
        self.kurtosis = None
        self.skew = None
        self.show_object_fit = False

    def __repr__(self):
        '''x y Flux  FWHM PeakSNR RMS Skew Kurtosis'''
    def output_data(self):
        return [self.peak_point[0], self.peak_point[1], self.cumulated_flux, "lul", "lul", self.rms, self.skew, self.kurtosis]

    def add_header_data( self, header_data ):
        self.header_data = header_data

    def add_background_data( self, background_data ):
        self.background_data = background_data

    def gaussian(self, data_tuple, height, center_x, center_y, width_x, width_y):
        """Returns a gaussian function with the given parameters"""
        (x, y) = data_tuple
        width_x = float(width_x)
        width_y = float(width_y)
        res = height*np.exp(-(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)
        return res.ravel()

    def twoD_Gaussian(self, data_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
        (x, y) = data_tuple
        xo = float(xo)
        yo = float(yo)
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
        return g.ravel()

    def veres(self, data_tuple, x0, y0, length, width, total_flux, rotation):
        (x, y) = data_tuple
        x0 = float(x0)
        y0 = float(y0)
        res = np.exp(-(((x-x0)*math.sin(math.radians(rotation)) + (y-y0)*math.cos(math.radians(rotation)))**2/(2*(width**2))))
        # res = self.background_data[y][x] +\
        #     (total_flux/length) *\
        #     (1/(2*width*math.sqrt(2*math.pi))) *\
        #     np.exp(-(((x-x0)*math.sin(math.radians(rotation)) + (y-y0)*math.cos(math.radians(rotation)))**2/(2*(width**2)))) *\
            # (\
            #     erf( ((x-x0)*math.cos(math.radians(rotation)) + (y-y0)*math.sin(math.radians(rotation) + (length/2) ))/(width*math.sqrt(2)) ) -\
            #     erf( ((x-x0)*math.cos(math.radians(rotation)) + (y-y0)*math.sin(math.radians(rotation) + (length/2) ))/(width*math.sqrt(2)) )\
            # )
        return res.ravel()

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

    def fill_to_square(self, square_width, square_height):
        max_value = 0
        for point in self.points:
            if self.image[point[1],point[0]] > max_value:
                self.peak_point = point
                max_value = self.image[point[1],point[0]]
        square = np.zeros((square_height, square_width))
        mid_x, mid_y = square_width // 2, square_height // 2
        for y in range(square_height):
            for x in range(square_width):
                relative_x = self.peak_point[0] - mid_x + x
                relative_y = self.peak_point[1] - mid_y + y
                square[y,x] = self.image[relative_y,relative_x]
        return square


    def fit_curve(self, function='gauss', square_size=(11,11)):
        try:
            self.squared_data = self.fill_to_square(*square_size)
        except IndexError:
            raise IndexError("Border object, ignore")
        if function == 'gauss':
            x = np.linspace(0, square_size[0]-1, square_size[0])
            y = np.linspace(0, square_size[1]-1, square_size[1])
            x, y = np.meshgrid(x, y)

            popt, pcov = curve_fit(self.twoD_Gaussian, (x, y), self.squared_data.flatten(), maxfev=500000000, xtol=1e-15, ftol=1e-15)

            self.predicted = self.twoD_Gaussian((x, y), *popt).reshape(*self.squared_data.shape)
            self.rms_res = rms(self.squared_data, self.predicted)
            if self.show_object_fit:
                print('==============')
                print('Root mean error:', self.rms_res)
                show_3d_data(self.squared_data, secondary_data=[self.predicted])

        if function == 'astropy_gauss':
            x = np.arange(0, self.squared_data.shape[1], 1)
            y = np.arange(0, self.squared_data.shape[0], 1)
            matrix_x, matrix_y = np.meshgrid(x, y)
            amp_init = np.matrix(self.squared_data).max()
            halfsize = 5
            stdev_init = 0.33 * halfsize

            def tie_stddev(model):  # we need this for tying x_std and y_std
                xstddev = model.x_stddev
                return xstddev

            params = self.moments(self.squared_data)
            t_init = models.Gaussian2D(x_mean=halfsize + 0.5, y_mean=halfsize + 0.5, x_stddev=stdev_init,
                                               y_stddev=stdev_init, amplitude=amp_init, tied={'y_stddev': tie_stddev})

            fit_m = fitting.LevMarLSQFitter()
            m = fit_m(t_init, matrix_x, matrix_y, self.squared_data)
            print(fit_m.fit_info['message'])

            self.predicted = np.zeros(self.squared_data.shape, dtype=int)
            for y, row in enumerate(self.squared_data):
                for x, val in enumerate(row):
                    self.predicted[y][x] = m(x,y)

            self.rms_res = rms(self.squared_data, self.predicted)

            print('Root mean error:', self.rms_res)


            show_3d_data(self.squared_data, secondary_data=[self.predicted])
        elif function == 'veres':
            self.length = 50 # from self.header_data
            self.width = 10 # from self.header_data
            self.rotation = 5 # rotation from self.header_data
            print('################')
            x = np.linspace(0, square_size[0]-1, square_size[0])
            y = np.linspace(0, square_size[1]-1, square_size[1])
            x, y = np.meshgrid(x, y)

            popt, pcov = curve_fit(self.veres, np.array((x, y), dtype=int), self.squared_data.flatten(), maxfev=500000000)
            print(pcov)

            self.predicted = self.veres((x, y), *popt).reshape(*self.squared_data.shape)
            if self.show_object_fit:
                print('==============')
                print('Root mean error:', rms(self.squared_data, self.predicted))
                show_3d_data(self.squared_data, secondary_data=[self.predicted])

        self.cumulated_flux = round(self.squared_data.sum())
        self.kurtosis = round(kurtosis(self.squared_data.flatten()), 3)
        self.skew = round(skew(self.squared_data.flatten()), 3)
        self.rms = round(self.rms_res, 3)
