import numpy as np


def dist_with_miss(a, b, l=0.0):
    if(len(a) != len(b)):
        return np.inf
    ls = l * np.ones(len(a))
    msk = ~ (np.isnan(a) | np.isnan(b))
    res = np.sum((np.abs(a-b)[msk]))+np.sum((ls[~msk]))
    return res
