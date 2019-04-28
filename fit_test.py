import numpy as np
from scipy.special import erf
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def erfunc(x, mFL, a, b):
    return mFL*erf((x-a)/(b*np.sqrt(2)))

x_data  = np.linspace(-3000, 3000, 100)

mFL, a, b = 0.0003, 500, 100

y_data  = erfunc(x_data, mFL, a, b)
y_noise = np.random.rand(y_data.size) / 1e4
y_noisy_data = y_data + y_noise

superior_params, extras = curve_fit(erfunc, x_data, y_noisy_data,
                                    p0=[0.001, 100, 100])

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)


ax2.plot(x_data, erfunc(x_data, *superior_params))
ax2.plot(x_data, y_noisy_data, 'k')
ax2.set_title('After Guesses')
plt.show()
