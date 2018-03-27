import numpy as np


class PointSpreadMesh(object):

    def __init__(self, points):
        self.points = points
        self.biggest_intensity_point = None
        self.horizontal_cross = list()
        self.vertical_cross = list()

    def get_point_crosses(self):
        raise NonImplementedError
