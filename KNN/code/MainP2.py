import time
import numpy as np
import KNN as knn
import ReadCSV as rcsv
import Shuffle as sf
import pandas as pd
import copy
import function

start_time = time.time()

data = np.array(rcsv.read_csv("energy_efficiency_data.csv"))

shuffled_data, shuffled_result_heating, shuffled_result_cooling = sf.shuffleP2(data)

# Removing the scientific notation
np.set_printoptions(suppress=True)
length_of_data = len(shuffled_data)

MAE_heating_table=pd.DataFrame(columns=["noUnweighted1","noUnweighted3","noUnweighted5","noUnweighted7","noUnweighted9"
                                     ,"noWeighted1","noWeighted3","noWeighted5","noWeighted7","noWeighted9"
                                     ,"NormalUnweighted1","NormalUnweighted3","NormalUnweighted5","NormalUnweighted7","NormalUnweighted9"
                                     ,"NormalWeighted1","NormalWeighted3","NormalWeighted5","NormalWeighted7","NormalWeighted9"],
                                index=["Fold1","Fold2", "Fold3", "Fold4","Fold5"])

MAE_cooling_table = copy.deepcopy(MAE_heating_table)

for i in range(5):
    # define start and end indexes for 5 fold
    start_index = int(length_of_data * 0.2 * i)
    end_index = int(length_of_data * 0.2 * (i + 1))

    # define train and test values
    X_train = np.concatenate((shuffled_data[0:start_index, :], shuffled_data[end_index:, :]))
    y_train_heating = np.concatenate((shuffled_result_heating[0:start_index], shuffled_result_heating[end_index:]))
    y_train_cooling = np.concatenate((shuffled_result_cooling[0:start_index], shuffled_result_cooling[end_index:]))
    X_test = shuffled_data[start_index:end_index, :]
    y_test_heating = shuffled_result_heating[start_index: end_index]
    y_test_cooling = shuffled_result_cooling[start_index: end_index]

    y_predict_heating, y_predict_cooling = knn.knn_algorithmP2(X_train, y_train_heating,y_train_cooling, X_test)
    for j in range(9,0,-2):
        # prints the scores
        column_name = "noUnweighted{}".format(j)
        row_name = "Fold{}".format(i + 1)
        MAE_heating = np.round((np.sum(np.abs(np.subtract(y_predict_heating[int((9-j)/2)],y_test_heating))) / len(y_test_heating)),decimals=3)
        MAE_heating_table[column_name][row_name] = MAE_heating
        MAE_cooling = np.round((np.sum(np.abs(np.subtract(y_predict_cooling[int((9-j)/2)],y_test_cooling))) / len(y_test_cooling)),decimals=3)
        MAE_cooling_table[column_name][row_name] = MAE_cooling

    for j in range(9,0,-2):
        column_name = "noWeighted{}".format(j)
        row_name = "Fold{}".format(i + 1)
        MAE_heating = np.round((np.sum(np.abs(np.subtract(y_predict_heating[int((19-j)/2)],y_test_heating))) / len(y_test_heating)),decimals=3)
        MAE_heating_table[column_name][row_name] = MAE_heating
        MAE_cooling = np.round((np.sum(np.abs(np.subtract(y_predict_cooling[int((19 - j) / 2)], y_test_cooling))) / len(y_test_cooling)),decimals=3)
        MAE_cooling_table[column_name][row_name] = MAE_cooling

#Normalization data
shuffled_data=function.normalization(shuffled_data)
for i in range(5):
    # define start and end indexes for 5 fold
    start_index = int(length_of_data * 0.2 * i)
    end_index = int(length_of_data * 0.2 * (i + 1))

    # define train and test values
    X_train = np.concatenate((shuffled_data[0:start_index, :], shuffled_data[end_index:, :]))
    y_train_heating = np.concatenate((shuffled_result_heating[0:start_index], shuffled_result_heating[end_index:]))
    y_train_cooling = np.concatenate((shuffled_result_cooling[0:start_index], shuffled_result_cooling[end_index:]))
    X_test = shuffled_data[start_index:end_index, :]
    y_test_heating = shuffled_result_heating[start_index: end_index]
    y_test_cooling = shuffled_result_cooling[start_index: end_index]

    y_predict_heating, y_predict_cooling = knn.knn_algorithmP2(X_train, y_train_heating,y_train_cooling, X_test)
    for j in range(9,0,-2):
        # prints the scores
        column_name = "NormalUnweighted{}".format(j)
        row_name = "Fold{}".format(i + 1)
        MAE_heating = np.round((np.sum(np.abs(np.subtract(y_predict_heating[int((9-j)/2)],y_test_heating))) / len(y_test_heating)),decimals=3)
        MAE_heating_table[column_name][row_name] = MAE_heating
        MAE_cooling = np.round((np.sum(np.abs(np.subtract(y_predict_cooling[int((9-j)/2)],y_test_cooling))) / len(y_test_cooling)),decimals=3)
        MAE_cooling_table[column_name][row_name] = MAE_cooling

    for j in range(9,0,-2):
        column_name = "NormalWeighted{}".format(j)
        row_name = "Fold{}".format(i + 1)
        MAE_heating = np.round((np.sum(np.abs(np.subtract(y_predict_heating[int((19-j)/2)],y_test_heating))) / len(y_test_heating)),decimals=3)
        MAE_heating_table[column_name][row_name] = MAE_heating
        MAE_cooling = np.round((np.sum(np.abs(np.subtract(y_predict_cooling[int((19 - j) / 2)], y_test_cooling))) / len(y_test_cooling)),decimals=3)
        MAE_cooling_table[column_name][row_name] = MAE_cooling

MAE_heating_table.to_csv("Heating_MAE.csv")
pd.DataFrame(MAE_heating_table.mean(axis=0)).T.round(decimals=3).to_csv("Average_Heating.csv")
MAE_cooling_table.to_csv("Cooling_MAE.csv")
pd.DataFrame(MAE_cooling_table.mean(axis=0)).T.round(decimals=3).to_csv("Average_Cooling.csv")

