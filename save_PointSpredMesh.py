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
        self.biggest_intensity_point = 0
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
                self.squared_data = self.fill_to_square(7,7)
            except IndexError:
                raise Exception("Border object, ignore")

        if function == 'gauss':
            params = self.moments(self.squared_data)
            errorfunction = lambda p: np.ravel(self.gaussian(*p)(*np.indices(self.squared_data.shape)) - self.squared_data)
            try:
                p, success = optimize.leastsq(errorfunction, params)
            except TypeError as e:
                return 'Error during fitting'
            print('#################################')
            print(self.squared_data)
            print(p)
            # print(self.gaussian(*p)(1,1))
            self.params = p
            return p
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
