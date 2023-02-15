import numpy as np


def read_csv(filename):

    """
    read the file and make it numpy array
    :param filename: is string of file path
    :return: numpy array
    """

    arr = np.loadtxt(filename, delimiter=",", dtype=str,  quotechar='"', )[1::, ::]
    return arr
