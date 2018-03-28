import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def show_3d_data(data, label='No Name'):
    # Set up grid and test data
    nx, ny = data.shape[1], data.shape[0]
    x = range(nx)
    y = range(ny)

    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')

    X, Y = np.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
    ha.plot_surface(X, Y, data)
    # ha.plot_wireframe(X, Y, data)
    # ha.contourf(X, Y, data)

    plt.show()

if __name__ == '__main__':
    show_3d_data(np.arange(16).reshape((4,4)))
