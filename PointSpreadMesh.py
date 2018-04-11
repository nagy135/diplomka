import numpy as np


class PointSpreadMesh(object):

    def __init__(self, points):
        self.points = points
        self.biggest_intensity_point = None
        self.horizontal_cross = list()
        self.vertical_cross = list()

    def get_point_crosses(self):
        raise NonImplementedError

    def gaus(x,a,x0,sigma):
        return a*exp(-(x-x0)**2/(2*sigma**2))

    def fit_curve():
        raise NonImplementedError
