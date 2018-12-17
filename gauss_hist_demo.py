import numpy as np
import pylab as plb
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
from astropy.io import fits

img = fits.open('result.fits')
flattened = img[0].data.flatten().astype(int)

# print(np.max(flattened))
# print(np.min(flattened))
# flattened += abs(np.min(flattened))
hist, bins = np.histogram(flattened, np.unique(flattened))

x = ar(range(len(bins) - 1))
# y = ar(flattened)
# assert False ,'die'
y = hist
# y = ar([0,0.5, 0.7, 1,1, 1,2,3,4,5,4,3,2,1, 1, 1, 1 ,0.6 ,0.5 ,0])

# n = len(x)                          #the number of data
# mean = sum(x*y)/n                   #note this correction
# sigma = sum(y*(x-mean)**2)/n        #note this correction

def gaus(x,a,x0,sigma):
    return a*exp(-(x-x0)**2/(2*sigma**2))

popt,pcov = curve_fit(gaus,x,y)
# print(popt)
# print(pcov)

plt.plot(x,y,'b:',label='data')
plt.plot(x,gaus(x,*popt),'r:',label='fit')
plt.legend()
plt.title('Fig. 3 - Fit for Time Constant')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.show()
