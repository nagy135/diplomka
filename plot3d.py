import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from time import gmtime, strftime
plotly.tools.set_credentials_file(username='nagy135', api_key='IOvZhqK5c8nhuu5QLpdr')

def show_3d_data(data, label='No Name', method='matplotlib'):
    if label == "No Name":
        label = 'Plot : {}'.format(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    if method == 'matplotlib':
        # Set up grid and test data
        nx, ny = data.shape[1], data.shape[0]
        x = range(nx)
        y = range(ny)

        hf = plt.figure()
        ha = hf.add_subplot(111, projection='3d')

        X, Y = np.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
        ha.view_init(0,90)
        ha.plot_surface(X, Y, data)
        # ha.plot_wireframe(X, Y, data)
        # ha.contourf(X, Y, data)

        plt.show()
        return
    if method == 'plotly':
        data = [
            go.Surface(
                z=data
            )
        ]
        layout = go.Layout(
            title=label,
            autosize=False,
            width=1000,
            height=1000,
            margin=dict(
                l=65,
                r=50,
                b=65,
                t=90
            )
        )
        fig = go.Figure(data=data, layout=layout)
        py.plot(fig, filename=label)

    else:
        raise Exception('Unknown plotting method')

if __name__ == '__main__':
    show_3d_data(np.arange(16).reshape((4,4)))
