import numpy as np


def mag2db(mag):
    return 20*np.log10(mag)
