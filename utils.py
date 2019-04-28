import numpy as np
import math
import sys

def normalize(data):
    return data/data.max()

def rms(data, predicted):
    return np.sqrt(
            np.sum(
                (normalize(predicted) - normalize(data))**2
                ) / predicted.size
            )

def neighbor_check(first_point, second_point):
    dist = np.linalg.norm( np.array(first_point) - np.array(second_point) )
    if dist == 1 or dist == math.sqrt(2):
        return True
    return False

def psnr(data, bg_median, noise_dispersion):
    peak = data.max()
    bg_median = float(bg_median)
    noise_dispersion = float(noise_dispersion)
    return round((peak-bg_median) / math.sqrt(peak - bg_median + noise_dispersion), 2)

def progressBar(value, total, bar_length=40):
    percent = float(value) / total
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\rPercent: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()
