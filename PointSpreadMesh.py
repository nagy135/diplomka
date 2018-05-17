import numpy as np
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
from scipy import optimize
from plot3d import *


class PointSpreadMesh(object):

    def __init__(self, points, image):
        self.points = points
        self.biggest_intensity_point = None
        self.squared_data = None
        self.image = image

    def gaussian(self, height, center_x, center_y, width_x, width_y):
        """Returns a gaussian function with the given parameters"""
        width_x = float(width_x)
        width_y = float(width_y)
        return lambda x,y: height*np.exp(
                    -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

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
        size_x, size_y = (max_x - min_x + 1, max_y - min_y + 1)
        square = np.zeros((size_y, size_x))
        for point in self.points:
            square[point[1]-min_y, point[0]-min_x] = self.image[point[1], point[0]]
        return square


    def fit_curve(self):
        """Returns (height, x, y, width_x, width_y)
        the gaussian parameters of a 2D distribution found by a fit"""
        if self.squared_data is None:
            data = self.fill_to_square()
        else:
            data = self.squared_data
        params = self.moments(data)
        errorfunction = lambda p: np.ravel(self.gaussian(*p)(*np.indices(data.shape)) - data)
        try:
            p, success = optimize.leastsq(errorfunction, params)
        except TypeError:
            return 'Error during fitting'
        return p

if __name__ == '__main__':
    instance = PointSpreadMesh(None)
    xin,yin = np.mgrid[0:201, 0:201]
    test_data = instance.gaussian(3,100,100,20,40)(xin, yin)
    point = PointSpreadMesh(test_data)
    print(point.fit_curve())
