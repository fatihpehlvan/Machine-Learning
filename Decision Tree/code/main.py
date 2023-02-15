from function import *
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from Tree import *
import sys

import os

sys.stdout = open(os.devnull, 'w')

# Read csv
data = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
data.head()

# Suffled data
data = data.sample(frac=1)

# Split data as train and test
X_train, X_test = train_test_split(data, train_size=0.8, shuffle=True)

# output test and train data
Y_train = X_train["Attrition"]
X_train = X_train.drop("Attrition", axis=1)
Y_test = X_test["Attrition"]
X_test = X_test.drop("Attrition", axis=1)

kf = KFold(n_splits=5)
validation=0
array_root=[]
mis_X=[]
mis_Y=[]
for train_index, test_index in kf.split(data):
    X_train = data.loc[train_index]
    X_test = data.loc[test_index]
    #Discretization process
    for attribution in X_train.columns:
        if X_train[attribution].dtype=="int64":
            X_train[attribution], X_test[attribution] = discretization(X_train[attribution].values,X_test[attribution].values, attribution)
    #output test and train data
    Y_train = X_train["Attrition"]
    X_train = X_train.drop("Attrition",axis=1)
    Y_test = X_test["Attrition"]
    X_test = X_test.drop("Attrition",axis=1)

    #generates tree
    root=generate_Tree(X_train,Y_train)

    array_root.append(root)

    #predict test values
    predict_array=[]
    for i in range(X_test.shape[0]):
        sample = X_test.iloc[i]
        predict_array.append(predict(root,sample))

    #Find some misclassified sample
    mis_index=np.where(Y_test!=predict_array)
    mis_index=np.random.choice(mis_index[0],3)
    mis_X.append(X_test.iloc[mis_index])
    mis_Y.append(Y_test.iloc[mis_index])

    validation+=1
    print("\nvalidation",validation)
    conf_matrix = confusion_matrix(y_true=Y_test, y_pred=predict_array)
    cm_display = ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = ["No", "Yes"])
    cm_display.plot(cmap=plt.cm.get_cmap("Blues"))
    plt.show()

    print("accuracy: " ,accuracy_score(Y_test, predict_array))
    print("precision: ", precision_score(Y_test,predict_array, pos_label="Yes"))
    print("recall: ",recall_score(Y_test,predict_array,pos_label="Yes"))
    print("f1_score: ", f1_score(Y_test,predict_array,pos_label="Yes"))

#print best decision tree model (5th)
root=array_root[4]
root.printTree()

# PART 2 STARTS !!!!!!!!!!!!!!!!!!!
sys.stdout = sys.__stdout__

X_train_p2, X_val, Y_train_p2, Y_val = train_test_split(X_train, Y_train, train_size=0.75, shuffle=True)

root_p2 = generate_Tree(X_train_p2, Y_train_p2)

predict_array = []
for i in range(X_test.shape[0]):
    sample = X_test.iloc[i]
    predict_array.append(predict(root_p2, sample))
print(accuracy_score(Y_test, predict_array))

find_leaves = []

root_p2.findLeaves(find_leaves)


def find_acc():
    predict_array = []
    for i in range(X_val.shape[0]):
        sample = X_val.iloc[i]
        if predict(root_p2, sample) is None:
            predict_array.append("No")
        else:
            predict_array.append(predict(root_p2, sample))
    return accuracy_score(Y_val, predict_array)


current_acc = find_acc()
print(current_acc)

removed_elements = []

while len(find_leaves) != 0:
    root_p2.findLeaves(find_leaves)
    find_leaves.sort()

    leaf = find_leaves.pop()
    leaf.isLeaf = True
    yes = leaf.numberofYes
    no = leaf.numberofNo
    leaf.predict = "No"
    if yes > no:
        leaf.predict = "Yes"
    new_acc = find_acc()

    if new_acc >= current_acc:
        removed_elements.append(leaf)
        current_acc = new_acc
        leaf.children.clear()
    else:
        leaf.predict = ""
        leaf.isLeaf = False
        break

root_p2.printTree()
predict_array = []
for i in range(X_test.shape[0]):
    sample = X_test.iloc[i]
    predict_array.append(predict(root_p2, sample))
print(accuracy_score(Y_test, predict_array))
