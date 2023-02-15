import numpy as np
import random
import copy
import pandas as pd


def findPerformanceMetric(prediction_array, original_array):
    """"
    findPerformanceMetric function calculate accuracy, precision and recall of your prediction array
    :param prediction_array: is a array, this is prediction value
    :param original_array: is a array, this is truth value
    :return: accuracy, average of precision  and average of recall
    """
    personality_list=np.unique(original_array)
    size = np.size(personality_list)
    cross_table = pd.DataFrame(data=np.zeros((size, size)), index=personality_list,
                               columns=personality_list,dtype=int)
    for i in range(np.size(prediction_array)):
        cross_table[original_array[i]][prediction_array[i]] +=1
    X = cross_table.values
    accuracy = round((np.sum(X.diagonal()) / np.sum(X)),3)
    precision = np.round((X.diagonal() / np.sum(X,axis=1)),decimals=3)
    recall = np.round((X.diagonal() / np.sum(X,axis=0)),decimals=3)
    cross_table['Precision'] = precision
    cross_table['Recall'] = recall
    cross_table['Accuracy'] = accuracy
    mean_precision = round(precision.mean(),3)
    mean_recall = round(recall.mean(),3)
    return accuracy, mean_precision, mean_recall, cross_table

def normalization(feature_array):
    """
    normalization function rescale each feature between (0-1) range
    :param feature_array:
    :return: normalization of feature array
    """
    feature_array = np.array(feature_array)
    coulumn_min=feature_array.min(axis=0)
    coulumn_max=feature_array.max(axis=0)
    feature_array=(feature_array-coulumn_min)/(coulumn_max-coulumn_min)
    return feature_array