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



class PointSpreadMesh(object):

    def __init__(self, points, image):
        self.points = points
        self.biggest_intensity_point = 0
        self.header_data = None
        self.background_data = None
        self.squared_data = None
        self.image = image
        self.kurtosis = None
        self.skew = None

    def add_header_data( self, header_data ):
        self.header_data = header_data

    def add_background_data( self, background_data ):
        self.background_data = background_data

    def gaussian(self, height, center_x, center_y, width_x, width_y):
        """Returns a gaussian function with the given parameters"""
        width_x = float(width_x)
        width_y = float(width_y)
        return lambda x,y: height*np.exp(-(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

    def twoD_Gaussian(self, data_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
        (x, y) = data_tuple
        xo = float(xo)
        yo = float(yo)
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
        return g.ravel()

    def veres(self, x0, y0, length, width, rotation):
        print('my params are {} {}'.format(x0, y0))
        return lambda x,y: self.background_data[y][x] +\
            (self.total_flux/length) *\
            (1/(2*width*math.sqrt(2*math.pi))) *\
            np.exp(-(((x-x0)*math.sin(math.radians(rotation)) + (y-y0)*math.cos(math.radians(rotation)))**2/(2*(width**2)))) *\
            (\
                erf( ((x-x0)*math.cos(math.radians(rotation)) + (y-y0)*math.sin(math.radians(rotation) + (length/2) ))/(width*math.sqrt(2)) ) -\
                erf( ((x-x0)*math.cos(math.radians(rotation)) + (y-y0)*math.sin(math.radians(rotation) + (length/2) ))/(width*math.sqrt(2)) )\
            )

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
                self.biggest_intensity_point = point
                max_value = self.image[point[1],point[0]]
        square = np.zeros((square_height, square_width))
        mid_x, mid_y = square_width // 2, square_height // 2
        for y in range(square_height):
            for x in range(square_width):
                relative_x = self.biggest_intensity_point[0] - mid_x + x
                relative_y = self.biggest_intensity_point[1] - mid_y + y
                square[y,x] = self.image[relative_y,relative_x]
        return square


    def fit_curve(self, function='gauss'):
        """Returns (height, x, y, width_x, width_y)
        the gaussian parameters of a 2D distribution found by a fit"""
        if self.squared_data is None:
            try:
                self.squared_data = self.fill_to_square(11,11)
            except IndexError:
                raise IndexError("Border object, ignore")

        if function == 'gauss':
            # params = self.moments(self.squared_data)
            # errorfunction = lambda p: np.ravel(self.gaussian(*p)(*np.indices(self.squared_data.shape)) - self.squared_data)
            # try:
            #     p, success = optimize.leastsq(errorfunction, params, maxfev=5000)
            #     p2, success = optimize.leastsq(errorfunction, params, maxfev=500000, xtol=1e-1)
            #     p3, success = optimize.leastsq(errorfunction, params, maxfev=50000000, ftol=1e-10)
            # except TypeError as e:
            #     return 'Error during fitting'
            # x = np.arange(0, self.squared_data.shape[1], 1)
            # y = np.arange(0, self.squared_data.shape[0], 1)
            # predicted = np.zeros(self.squared_data.shape)
            # predicted2 = np.zeros(self.squared_data.shape)
            # predicted3 = np.zeros(self.squared_data.shape)
            # for y, row in enumerate(self.squared_data):
            #     for x, val in enumerate(row):
            #         predicted[y][x] = self.gaussian(*p)(x,y)
            #         predicted2[y][x] = self.gaussian(*p2)(x,y)
            #         predicted3[y][x] = self.gaussian(*p3)(x,y)
            # show_3d_data(self.squared_data, secondary_data=[predicted, predicted2, predicted3])
            # print('#################################')
            # print('Root mean error:', rms(self.squared_data, predicted))
            # # print(self.gaussian(*p)(1,1))
            # self.params = p
            # return p
            x = np.linspace(0, 10, 11)
            y = np.linspace(0, 10, 11)
            x, y = np.meshgrid(x, y)

            # initial_guess = (3,100,100,20,40,0,10)
            popt, pcov = curve_fit(self.twoD_Gaussian, (x, y), self.squared_data.flatten(), maxfev=50000000)

            predicted = self.twoD_Gaussian((x, y), *popt).reshape(*self.squared_data.shape)
            print('Root mean error:', rms(self.squared_data, predicted))
            show_3d_data(self.squared_data, secondary_data=[predicted])

        if function == 'astropy_gauss':
            x = np.arange(0, self.squared_data.shape[1], 1)
            y = np.arange(0, self.squared_data.shape[0], 1)
            matrix_x, matrix_y = np.meshgrid(x, y)
            amp_init = np.matrix(self.squared_data).max()
            halfsize = 5
            stdev_init = 0.33 * halfsize

            # Fit the data using a box model.
            # Bounds are not really needed but included here to demonstrate usage.

            def tie_stddev(model):  # we need this for tying x_std and y_std
                xstddev = model.x_stddev
                return xstddev

            params = self.moments(self.squared_data)
            t_init = models.Gaussian2D(x_mean=halfsize + 0.5, y_mean=halfsize + 0.5, x_stddev=stdev_init,
                                               y_stddev=stdev_init, amplitude=amp_init, tied={'y_stddev': tie_stddev})

            fit_m = fitting.LevMarLSQFitter()
            m = fit_m(t_init, matrix_x, matrix_y, self.squared_data)
            print(fit_m.fit_info['message'])

            predicted = np.zeros(self.squared_data.shape, dtype=int)
            for y, row in enumerate(self.squared_data):
                for x, val in enumerate(row):
                    predicted[y][x] = m(x,y)

            rme = rms(self.squared_data, predicted)

            print('Root mean error:', rme)

            self.kurtosis = kurtosis(self.squared_data.flatten())
            self.skew = skew(self.squared_data.flatten())
            print('kurtosis: {}, skew: {}'.format(self.kurtosis, self.skew))

            show_3d_data(self.squared_data, secondary_data=[predicted])
        elif function == 'veres':
            self.length = 50 # from self.header_data
            self.width = 10 # from self.header_data
            self.total_flux = self.squared_data.sum() # from self.header_data
            self.rotation = 5 # rotation from self.header_data
            # x0 = math.ceil(np.mean((self.min_x, self.max_x)))
            # y0 = math.ceil(np.mean((self.min_y, self.max_y)))
            x0 = len(self.squared_data[0])//2
            y0 = len(self.squared_data)//2

            params = np.array([x0, y0, 50, 10, 5])
            print('fitting object with params {}'.format(params))
            # print(self.veres(*params)(2,2))
            print(self.squared_data)
            res = self.veres(*params)(2, 2)
            print(np.indices(self.squared_data.shape))
            print(res)
            assert False, 'end'
            errorfunction = lambda p: np.ravel(self.veres(*p)(*np.indices(self.squared_data.shape)))
            try:
                p, success = optimize.leastsq(errorfunction, params)
                assert False, 'SOMETHING WORKS'
            except TypeError as e:
                return 'Error during fitting'
            print('################')
            # print('fitting function to x0: {}, y0: {}'.format(x0, y0))
            # print(self.points)

if __name__ == '__main__':
    instance = PointSpreadMesh(None)
    xin,yin = np.mgrid[0:201, 0:201]
    test_data = instance.gaussian(3,100,100,20,40)(xin, yin)
    point = PointSpreadMesh(test_data)
    print(point.fit_curve())
