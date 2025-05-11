import numpy as np

def deadzone(x, a):
    return np.where(np.abs(x) <= a, 0, x - np.sign(x) * a)

def near_zero(arr, threshold=None):
    if threshold is None:
        threshold = 1e-6
    arr[np.abs(arr) < threshold] = 0
    return arr