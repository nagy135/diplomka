import numpy as np

def normalized(data):
    return data/data.max()

def rms(data, predicted):
    return np.sqrt(
            np.sum(
                (normalized(predicted) - normalized(data))**2
                ) / predicted.size
            )
