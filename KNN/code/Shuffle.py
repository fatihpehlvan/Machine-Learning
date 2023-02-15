import numpy as np


def shuffle(data):

    """
    shuffles the given data
    :param data: is a numpy array
    :return: shuffled data indexes, shuffled data attributes and shuffled data results
    """

    shuffled_data = np.zeros((10000, 62), dtype='<U116')

    sum = set({})
    for i in range(len(data)):
        sum.add(str(i))
    count = 0

    for i in sum:
        shuffled_data[count] = np.copy(data[int(i)])
        count += 1

    return shuffled_data[:, 0], shuffled_data[:, 1:-1].astype('float64'), shuffled_data[:, -1]


def shuffleP2 (data):
    """
    shuffles the given data
    :param data:
    :return:
    """
    shuffled_data = np.zeros((768, 10), dtype=float)

    sum = set({})
    for i in range(len(data)):
        sum.add(str(i))
    count = 0
    for i in sum:
        shuffled_data[count] = np.copy(data[int(i)])
        count += 1
    return shuffled_data[:, :-2].astype('float64'), shuffled_data[:, -2].astype('float64'), shuffled_data[:, -1].astype('float64')
