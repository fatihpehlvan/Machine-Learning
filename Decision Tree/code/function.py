import numpy as np
import pandas as pd
import math
from Tree import *


def discretization(train_attribute_values, test_attribute_values, attribute_name):
    """
    Discretize continues variable
    :param train_attribute_values: array of train attribute value
    :param test_attribute_values: array of test attribute value
    :param attribute_name:
    :return:
    """

    # Find first quartile, second quartile and third quartile of attribute values
    Q1 = np.percentile(train_attribute_values, 25)
    Q2 = np.percentile(train_attribute_values, 50)
    Q3 = np.percentile(train_attribute_values, 75)

    # train attribute value < Q1
    index1 = np.where(train_attribute_values <= Q1)
    index2 = np.where((train_attribute_values > Q1) & (train_attribute_values <= Q2))
    index3 = np.where((train_attribute_values > Q2) & (train_attribute_values <= Q3))
    index4 = np.where((train_attribute_values > Q3))

    # test attribute value index for restricted area
    test_index1 = np.where(test_attribute_values <= Q1)
    test_index2 = np.where((test_attribute_values > Q1) & (test_attribute_values <= Q2))
    test_index3 = np.where((test_attribute_values > Q2) & (test_attribute_values <= Q3))
    test_index4 = np.where((test_attribute_values > Q3))

    test_attribute_values = test_attribute_values.astype("<U33")
    train_attribute_values = train_attribute_values.astype("<U33")
    for i, j, k, l in zip(index1, index2, index3, index4):
        train_attribute_values[i] = "{}<={}".format(attribute_name, Q1)
        train_attribute_values[j] = "{}<{}<={}".format(Q1, attribute_name, Q2)
        train_attribute_values[k] = "{}<{}<={}".format(Q2, attribute_name, Q3)
        train_attribute_values[l] = "{}<{}".format(Q3, attribute_name)

    for i, j, k, l in zip(test_index1, test_index2, test_index3, test_index4):
        test_attribute_values[i] = "{}<={}".format(attribute_name, Q1)
        test_attribute_values[j] = "{}<{}<={}".format(Q1, attribute_name, Q2)
        test_attribute_values[k] = "{}<{}<={}".format(Q2, attribute_name, Q3)
        test_attribute_values[l] = "{}<{}".format(Q3, attribute_name)

    return train_attribute_values, test_attribute_values


def cal_total_data_entropy(output_values):
    """
    calculate total data entropy
    :param output_values: array of train output data (in this assignment it is Attrition values")
    :return:
    """
    output_unique, counts_elements = np.unique(output_values, return_counts=True)
    entropy = 0
    for count in counts_elements:
        prob = count / sum(counts_elements)
        entropy -= prob * math.log2(prob)
    return entropy


def cal_attribute_entropy(attribute_values, output_values):
    """
    calculate attribution entropy
    :param attribute_values: array of train attribution data
    :param output_values: array of train output data
    :return:
    """
    attribute_unique_values = np.unique(attribute_values)
    output_unique_values = np.unique(output_values)
    data_set = np.column_stack((attribute_values, output_values))
    total_entropy = 0
    for i in attribute_unique_values:
        sub1 = data_set[np.where(data_set[:, 0] == i)]
        entropy = 0
        for j in output_unique_values:
            sub2 = sub1[np.where(sub1[:, 1] == j)]
            prob = np.size(sub2, axis=0) / np.size(sub1, axis=0)
            if prob != 0:
                entropy -= prob * math.log2(prob)
        total_entropy += np.size(sub1, axis=0) / np.size(data_set, axis=0) * entropy
    return total_entropy


def cal_gain(train_attribute_data, train_output_data):
    """
    calculate information gain for a feature
    :param train_attribute_data: array of train attribution data
    :param train_output_data: array of train output data
    :return:
    """
    return cal_total_data_entropy(train_output_data) - cal_attribute_entropy(train_attribute_data, train_output_data)


def find_most_gain_attribute(X_train, Y_train):
    """
    calculate feature that has most information gain
    :param X_train: all train data except output
    :param Y_train: train output data
    :return:
    """
    max_gain = 0
    attribute_name = ""
    for attribute in X_train.columns:
        attribute_value = X_train[attribute].values
        attribute_gain = cal_gain(attribute_value, Y_train)
        # print("{} = {}".format(attribute,attribute_gain))
        if attribute_gain > max_gain:
            max_gain = attribute_gain
            attribute_name = attribute
    return attribute_name


def generate_Tree(X_train, Y_train):
    """
    #Generates decision tree
    :param X_train: current train data
    :param Y_train: current output data
    :return:
    """
    #create new Node
    root = Node()
    #find max gain attribute
    max_feature=find_most_gain_attribute(X_train,Y_train)
    #find unique attribute values
    output_label, output_count = np.unique(Y_train,return_counts=True)
    #feature name is root value
    root.value=max_feature
    root.gain=cal_gain(X_train[max_feature],Y_train)
    for i,j in zip(output_label, output_count):
        if i=="Yes":
            root.numberofYes = j
        else:
            root.numberofNo=j

    #branc of root are unique values of feature
    unique_val, counts = np.unique(X_train[max_feature], return_counts=True)

    for label, count in zip(unique_val,counts):
        index = X_train[max_feature]==label
        X_train_sub=X_train[index]
        Y_train_sub=Y_train[index]

        output_label, output_count = np.unique(Y_train_sub,return_counts=True)

        # If attribute entropy is 0, this label is pure node
        if cal_attribute_entropy(X_train_sub[max_feature],Y_train_sub)==0:
            leafNode = Node()
            leafNode.value=label
            for i,j in zip(output_label, output_count):
                if i=="Yes":
                    leafNode.numberofYes = j
                else:
                    leafNode.numberofNo=j
            leafNode.isLeaf=True
            leafNode.predict = Y_train_sub.values[0]
            root.insertChild(leafNode)

        #If attribute entropy is not 0, this label is new root.
        else:
            newRoot = Node()
            newRoot.value=label
            for i,j in zip(output_label, output_count):
                if i=="Yes":
                    newRoot.numberofYes = j
                else:
                    newRoot.numberofNo = j
            X_train_sub_copy = X_train_sub.drop(max_feature,axis=1)
            #Find child root recursively
            child = generate_Tree(X_train_sub_copy,Y_train_sub)
            newRoot.insertChild(child)
            root.insertChild(newRoot)
    return root

#Predict output of a sample based on decision tree model
def predict(root, sample):
    for child in root.children:
        if child.value == sample[root.value]:
            if child.isLeaf:
                return child.predict
            else:
                return predict (child.children[0], sample)
    if root.numberofYes > root.numberofNo:
        return "Yes"
    else:
        return "No"
