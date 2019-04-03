import numpy as np
import math
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
from scipy import optimize
from scipy.special import erf
from plot3d import *


class PointSpreadMesh(object):

    def __init__(self, points, image):
        self.points = points
        self.biggest_intensity_point = None
        self.header_data = None
        self.background_data = None
        self.squared_data = None
        self.image = image

    def add_header_data( self, header_data ):
        self.header_data = header_data

    def add_background_data( self, background_data ):
        self.background_data = background_data

    def gaussian(self, height, center_x, center_y, width_x, width_y):
        """Returns a gaussian function with the given parameters"""
        width_x = float(width_x)
        width_y = float(width_y)
        return lambda x,y: height*np.exp(-(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

    def veres(self, x0, y0, total_flux, length, width, rotation):
        print('my params are {} {} {} {} {} {}'.format(x0, y0, total_flux, length, width, rotation))
        # return lambda x,y: x+y
        return lambda x,y: self.background_data[y][x] + (total_flux/length)*(1/(2*width*math.sqrt(2*math.pi))) * np.exp(-(((x-x0)*math.sin(math.radians(rotation)) + (y-y0)*math.cos(math.radians(rotation)))**2/(2*(width**2)))) * ( erf( ((x-x0)*math.cos(math.radians(rotation)) + (y-y0)*math.sin(math.radians(rotation) + (length/2) ))/(width*math.sqrt(2)) ) - erf( ((x-x0)*math.cos(math.radians(rotation)) + (y-y0)*math.sin(math.radians(rotation) + (length/2) ))/(width*math.sqrt(2)) ) )

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

    def fill_to_square(self):
        min_x, max_x, min_y, max_y = None, None, None, None
        for point in self.points:
            if min_x is None or min_x > point[0]:
                min_x = point[0]
            if max_x is None or max_x < point[0]:
                max_x = point[0]
            if min_y is None or min_y > point[1]:
                min_y = point[1]
            if max_y is None or max_y < point[1]:
                max_y = point[1]
        self.min_x, self.max_x, self.min_y, self.max_y = min_x, max_x, min_y, max_y
        size_x, size_y = (max_x - min_x + 1, max_y - min_y + 1)
        square = np.zeros((size_y, size_x))
        shaped_indices = np.zeros((size_y, size_x,2))
        for point in self.points:
            square[point[1]-min_y, point[0]-min_x] = self.image[point[1], point[0]]
            shaped_indices[point[1]-min_y, point[0]-min_x] = np.array([point[1], point[0]])
        for y in range(square.shape[0]):
            for x in range(square.shape[1]):
                if square[y][x] == .0:
                    square[y][x] = self.image[min_y+y][min_x+x]
        self.shaped_indices = shaped_indices
        return square


    def fit_curve(self, function='gauss'):
        """Returns (height, x, y, width_x, width_y)
        the gaussian parameters of a 2D distribution found by a fit"""
        if self.squared_data is None:
            self.squared_data = self.fill_to_square()

        if function == 'gauss':
            params = self.moments(self.squared_data)
            errorfunction = lambda p: np.ravel(self.gaussian(*p)(*np.indices(self.squared_data.shape)) - self.squared_data)
            try:
                p, success = optimize.leastsq(errorfunction, params)
            except TypeError as e:
                return 'Error during fitting'
            print('#################################')
            print(self.squared_data)
            print(self.points)
            print(p)
            # print(self.gaussian(*p)(1,1))
            self.params = p
            return p
        elif function == 'veres':
            length = 50 # from self.header_data
            width = 10 # from self.header_data
            total_flux = self.squared_data.sum() # from self.header_data
            rotation = 5 # rotation from self.header_data
            # x0 = math.ceil(np.mean((self.min_x, self.max_x)))
            # y0 = math.ceil(np.mean((self.min_y, self.max_y)))
            x0 = len(self.squared_data[0])//2
            y0 = len(self.squared_data)//2

            params = (x0, y0, total_flux, length, width, rotation)
            print('fitting object with params {}'.format(params))
            print(self.veres(*params)(2,2))
            print(self.squared_data)
            assert False, 'trolllllls'
            # errorfunction = lambda p: np.ravel(self.veres(*p)(*np.indices(self.squared_data.shape)) - self.squared_data)
            # try:
            #     p, success = optimize.leastsq(errorfunction, params)
            #     assert False, 'SOMETHING WORKS'
            # except TypeError as e:
            #     return 'Error during fitting'
            print('################')
            print('fitting function to x0: {}, y0: {}'.format(x0, y0))
            print(self.points)
            assert False

if __name__ == '__main__':
    instance = PointSpreadMesh(None)
    xin,yin = np.mgrid[0:201, 0:201]
    test_data = instance.gaussian(3,100,100,20,40)(xin, yin)
    point = PointSpreadMesh(test_data)
    print(point.fit_curve())
