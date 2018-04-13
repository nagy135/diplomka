import numpy as np
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp


class PointSpreadMesh(object):

    def __init__(self, points):
        self.points = points
        self.biggest_intensity_point = None
        self.horizontal_cross = list()
        self.vertical_cross = list()
        self.popt = None
        self.pcov = None

    def get_point_crosses(self):
        raise NonImplementedError

    def gaus(x,a,x0,sigma):
        return a*exp(-(x-x0)**2/(2*sigma**2))

    def fit_curve():
        self.get_point_crosses()
        self.horizontal_popt, self.horizontal_pcov = self._inner_fit_curve(np.arange(len(self.horizontal_cross)), self.horizontal_cross)
        self.vertical_popt, self.vertical_pcov = self._inner_fit_curve(np.arange(len(self.vertical_cross)), self.vertical_cross)


    def _inner_fit_curve(x, y):
        n = len(x)
        mean = sum(x*y)/n
        sigma = sum(y*(x-mean)**2)/n

        popt, pcov = curve_fit(self.gaus,x,y,p0=[1,mean,sigma])
        return popt, pcov
