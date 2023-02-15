import numpy as np
import sys


class Attribute:
    def __init__(self, name, distance):
        self.name = name
        self.distance = distance

    def __lt__(self, other):
        return self.distance < other.distance

    def getName(self):
        return self.name

    def getDistance(self):
        return self.distance

    def getWeight(self):
        return 1 / self.distance

    def __str__(self):
        return str(self.name) + " " + str(self.distance)


def find_max(numpy_arr):
    """
    find_max function find the maximum value, and it is index
    :param numpy_arr: contains Attribute objects
    :return: maximum value, and it is index
    """

    max = numpy_arr[0]
    count = 0
    index = 0
    for i in numpy_arr:
        if i.getDistance() > max.getDistance():
            max = i
            index = count
        count += 1
    return max, index


def most_frequent(numpy_arr, weighted):
    """
    find the maximum weight of given array's attributes
    :param numpy_arr: contains Attribute objects
    :param weighted: is a boolean value. Determine the calculation is weighted or not
    :return: maximum weight of Attribute object
    """

    dict = {}
    for i in numpy_arr:
        if dict.get(i.getName()) is None:
            if weighted:
                dict[i.getName()] = i.getWeight()
            else:
                dict[i.getName()] = 1
        else:
            if weighted:
                dict[i.getName()] = dict.get(i.getName()) + i.getWeight()
            else:
                dict[i.getName()] = dict.get(i.getName()) + 1
    return max(dict, key=dict.get)


def most_frequentP2(numpy_arr, weighted):
    """
    find the maximum weight of given array's attributes
    :param numpy_arr: contains Attribute objects
    :param weighted: is a boolean value. Determine the calculation is weighted or not
    :return: maximum weight of Attribute object
    """

    if weighted:
        numerator = 0
        denominator = 0
        for i in numpy_arr:
            numerator += (i.getName() * i.getWeight())
            denominator += i.getWeight()
        if denominator == 0:
            denominator = 0.00001
        return numerator / denominator
    else:
        sum = 0
        for i in numpy_arr:
            sum += i.getName()
        return sum / len(numpy_arr)


def knn_algorithm(X_train, y_train, X_test):
    """
    knn_algorithm function calculate 9, 7, 5, 3, 1 nearest neighbor for weighted and unweighted according to given params
    :param X_train: is a numpy array, this is train attribute values
    :param y_train: is a numpy array, this is train results
    :param X_test: is a numpy array, this is test attribute values
    :return: a numpy_array list, for k = 9, 7, 5, 3 ,1 respectively
    """

    min_numbers = np.full((9), Attribute("", sys.maxsize), dtype=Attribute)
    returnArray = np.zeros((10, len(X_test)), dtype='<U6')
    arrayCounter = 0
    for i in X_test:
        for k in range(9):
            min_numbers[k] = Attribute("", sys.maxsize)
        max_num = min_numbers[0]
        index = 0
        count = 0
        for j in X_train:
            # calculete the distance
            arr = np.subtract(i, j)
            arr = np.power(arr, 2)
            sum = np.sum(arr) ** 0.5
            if sum < max_num.getDistance():
                min_numbers[index] = Attribute(y_train[count], sum)
            max_num, index = find_max(min_numbers)
            count += 1
        np.sort(min_numbers)
        returnArray[0][arrayCounter] = most_frequent(min_numbers, False)
        returnArray[1][arrayCounter] = most_frequent(min_numbers[: -2], False)
        returnArray[2][arrayCounter] = most_frequent(min_numbers[: -4], False)
        returnArray[3][arrayCounter] = most_frequent(min_numbers[: -6], False)
        returnArray[4][arrayCounter] = most_frequent(min_numbers[: -8], False)
        returnArray[5][arrayCounter] = most_frequent(min_numbers, True)
        returnArray[6][arrayCounter] = most_frequent(min_numbers[: -2], True)
        returnArray[7][arrayCounter] = most_frequent(min_numbers[: -4], True)
        returnArray[8][arrayCounter] = most_frequent(min_numbers[: -6], True)
        returnArray[9][arrayCounter] = most_frequent(min_numbers[: -8], True)

        arrayCounter += 1

    return returnArray


def knn_algorithmP2(X_train, y_train_heating, y_traing_cooling, X_test):
    # distance / value
    min_numbers_heating = np.full((9), Attribute(0, sys.maxsize), dtype=Attribute)
    min_numbers_cooling = np.full((9), Attribute(0, sys.maxsize), dtype=Attribute)
    returnArrayHeating = np.zeros((10, len(X_test)), dtype=float)
    returnArrayCooling = np.zeros((10, len(X_test)), dtype=float)
    arrayCounter = 0
    for i in X_test:
        for k in range(9):
            min_numbers_heating[k] = Attribute(0, sys.maxsize)
            min_numbers_cooling[k] = Attribute(0, sys.maxsize)
        max_num_heating = min_numbers_heating[0]
        index_heating = 0
        max_num_cooling = min_numbers_cooling[0]
        index_cooling = 0
        count = 0
        for j in X_train:
            # calculete the distance
            arr = np.subtract(i, j)
            arr = np.power(arr, 2)
            sum = np.sum(arr) ** 0.5
            if sum < max_num_heating.getDistance():
                min_numbers_heating[index_heating] = Attribute(y_train_heating[count], sum)
            max_num_heating, index_heating = find_max(min_numbers_heating)
            if sum < max_num_cooling.getDistance():
                min_numbers_cooling[index_cooling] = Attribute(y_traing_cooling[count], sum)
            max_num_cooling, index_cooling = find_max(min_numbers_cooling)
            count += 1

        np.sort(min_numbers_heating)
        np.sort(min_numbers_cooling)
        returnArrayHeating[0][arrayCounter] = most_frequentP2(min_numbers_heating, False)
        returnArrayHeating[1][arrayCounter] = most_frequentP2(min_numbers_heating[: -2], False)
        returnArrayHeating[2][arrayCounter] = most_frequentP2(min_numbers_heating[: -4], False)
        returnArrayHeating[3][arrayCounter] = most_frequentP2(min_numbers_heating[: -6], False)
        returnArrayHeating[4][arrayCounter] = most_frequentP2(min_numbers_heating[: -8], False)
        returnArrayHeating[5][arrayCounter] = most_frequentP2(min_numbers_heating, True)
        returnArrayHeating[6][arrayCounter] = most_frequentP2(min_numbers_heating[: -2], True)
        returnArrayHeating[7][arrayCounter] = most_frequentP2(min_numbers_heating[: -4], True)
        returnArrayHeating[8][arrayCounter] = most_frequentP2(min_numbers_heating[: -6], True)
        returnArrayHeating[9][arrayCounter] = most_frequentP2(min_numbers_heating[: -8], True)

        returnArrayCooling[0][arrayCounter] = most_frequentP2(min_numbers_cooling, False)
        returnArrayCooling[1][arrayCounter] = most_frequentP2(min_numbers_cooling[: -2], False)
        returnArrayCooling[2][arrayCounter] = most_frequentP2(min_numbers_cooling[: -4], False)
        returnArrayCooling[3][arrayCounter] = most_frequentP2(min_numbers_cooling[: -6], False)
        returnArrayCooling[4][arrayCounter] = most_frequentP2(min_numbers_cooling[: -8], False)
        returnArrayCooling[5][arrayCounter] = most_frequentP2(min_numbers_cooling, True)
        returnArrayCooling[6][arrayCounter] = most_frequentP2(min_numbers_cooling[: -2], True)
        returnArrayCooling[7][arrayCounter] = most_frequentP2(min_numbers_cooling[: -4], True)
        returnArrayCooling[8][arrayCounter] = most_frequentP2(min_numbers_cooling[: -6], True)
        returnArrayCooling[9][arrayCounter] = most_frequentP2(min_numbers_cooling[: -8], True)

        arrayCounter += 1
    return returnArrayHeating, returnArrayCooling
