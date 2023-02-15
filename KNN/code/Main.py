import copy
import os

import Shuffle as sf
import ReadCSV as rcsv
import numpy as np
import KNN as knn
import function
import pandas as pd

data = np.array(rcsv.read_csv("subset_16P.csv"))


# Shuffle
shuffled_indexes, shuffled_data, shuffled_result_data = sf.shuffle(data)

length_of_data = len(shuffled_data)


"""
Result
nonNormalized Unweighted = noUnweighted
nonNormalized Weighted = noWeighted
Normalized Unweighted = NormalUnweighted
Normalized Weighted = NormalWeighted 
"""

accuracy_table=pd.DataFrame(columns=["noUnweighted1","noUnweighted3","noUnweighted5","noUnweighted7","noUnweighted9"
                                     ,"noWeighted1","noWeighted3","noWeighted5","noWeighted7","noWeighted9"
                                     ,"NormalUnweighted1","NormalUnweighted3","NormalUnweighted5","NormalUnweighted7","NormalUnweighted9"
                                     ,"NormalWeighted1","NormalWeighted3","NormalWeighted5","NormalWeighted7","NormalWeighted9"],
                            index=["Fold1","Fold2", "Fold3", "Fold4","Fold5"])

recall_table=copy.deepcopy(accuracy_table)
precision_table=copy.deepcopy(accuracy_table)


# K-NN Algorithm
os.makedirs("nonNormalizedUnweightOutput", exist_ok=True)
os.makedirs("nonNormalizedWeightedOutput", exist_ok=True)

for i in range(5):
    # define start and end indexes for 5 fold
    start_index = int(length_of_data * 0.2 * i)
    end_index = int(length_of_data * 0.2 * (i + 1))

    # define train and test values
    X_train = np.concatenate((shuffled_data[0:start_index, :], shuffled_data[end_index:, :]))
    y_train = np.concatenate((shuffled_result_data[0:start_index], shuffled_result_data[end_index:]))
    X_test = shuffled_data[start_index:end_index, :]
    y_test = shuffled_result_data[start_index: end_index]

    # knn_algorithm return a numpy_array list, for k = 9, 7, 5, 3 ,1 respectively
    y_predict = knn.knn_algorithm(X_train, y_train, X_test)

    #result of unweighted part
    for j in range(9,0,-2):
        # prints the scores
        column_name = "noUnweighted{}".format(j)
        row_name = "Fold{}".format(i+1)
        file_name = "nonNormalizedUnweightOutput/Fold{}k{}.csv"
        file_name = file_name.format(i+1,j)
        accuracy, precision, recall, cross_table = function.findPerformanceMetric(y_predict[int((9-j)/2)],y_test)
        cross_table.to_csv(file_name)
        accuracy_table[column_name][row_name] = accuracy
        precision_table[column_name][row_name] = precision
        recall_table[column_name][row_name] = recall


    #result of weighted part
    for j in range(9,0,-2):
        column_name = "noWeighted{}".format(j)
        row_name = "Fold{}".format(i + 1)
        file_name = "nonNormalizedWeightedOutput/Fold{}k{}.csv"
        file_name = file_name.format(i+1, j)
        accuracy, precision, recall, cross_table = function.findPerformanceMetric(y_predict[int((19-j)/2)], y_test)
        cross_table.to_csv(file_name)
        accuracy_table[column_name][row_name] = accuracy
        precision_table[column_name][row_name] = precision
        recall_table[column_name][row_name] = recall




#normalized K-NN Algorithm
shuffled_data=function.normalization(shuffled_data)
os.makedirs("NormalizedUnweightOutput", exist_ok=True)
os.makedirs("NormalizedWeightedOutput", exist_ok=True)

for i in range(5):
    # define start and end indexes for 5 fold
    start_index = int(length_of_data * 0.2 * i)
    end_index = int(length_of_data * 0.2 * (i + 1))

    # define train and test values
    X_train = np.concatenate((shuffled_data[0:start_index, :], shuffled_data[end_index:, :]))
    y_train = np.concatenate((shuffled_result_data[0:start_index], shuffled_result_data[end_index:]))
    X_test = shuffled_data[start_index:end_index, :]
    y_test = shuffled_result_data[start_index: end_index]

    # knn_algorithm return a numpy_array list, for k = 9, 7, 5, 3 ,1 respectively
    y_predict = knn.knn_algorithm(X_train, y_train, X_test)

    #result of unweighted
    for j in range(9,0,-2):
        # prints the scores
        column_name = "NormalUnweighted{}".format(j)
        row_name = "Fold{}".format(i + 1)
        file_name = "NormalizedUnweightOutput/NorUnweightFold{}k{}.csv"
        file_name = file_name.format(i+1,j)
        accuracy, precision, recall, cross_table = function.findPerformanceMetric(y_predict[int((9-j)/2)],y_test)
        cross_table.to_csv(file_name)
        accuracy_table[column_name][row_name] = accuracy
        precision_table[column_name][row_name] = precision
        recall_table[column_name][row_name] = recall

    #result of weighted
    for j in range(9,0,-2):
        # prints the scores
        column_name = "NormalWeighted{}".format(j)
        row_name = "Fold{}".format(i + 1)
        file_name = "NormalizedWeightedOutput/NorWeightFold{}k{}.csv"
        file_name = file_name.format(i+1,j)
        accuracy, precision, recall, cross_table = function.findPerformanceMetric(y_predict[int((19-j)/2)],y_test)
        cross_table.to_csv(file_name)
        accuracy_table[column_name][row_name] = accuracy
        precision_table[column_name][row_name] = precision
        recall_table[column_name][row_name] = recall

accuracy_table.to_csv("accuracy.csv")
precision_table.to_csv("precision.csv")
recall_table.to_csv("recall.csv")

mean_accuracy = pd.DataFrame(accuracy_table.mean(axis=0)).T
mean_accuracy.to_csv("mean_accuracy.csv")
mean_precision = pd.DataFrame(precision_table.mean(axis=0)).T
mean_precision.to_csv("mean_precision.csv")
mean_recall=pd.DataFrame(recall_table.mean(axis=0)).T
mean_recall.to_csv("mean_recall.csv")