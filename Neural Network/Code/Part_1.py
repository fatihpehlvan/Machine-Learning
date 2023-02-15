import os

import numpy as np
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, recall_score, precision_score
from sklearn.utils import shuffle

epsilon = 10 ** -12
px = 60
# Read images
folders = np.array(["Bean", "Bitter_Gourd", "Bottle_Gourd", "Brinjal", "Broccoli", "Cabbage", "Capsicum", "Carrot",
                    "Cauliflower", "Cucumber", "Papaya", "Potato", "Pumpkin", "Radish", "Tomato"])


def load_images_from_folder(images, folder, count):
    for filename in os.listdir(folder):
        if any([filename.endswith(".jpg")]):
            img = np.asarray(Image.open(os.path.join(folder, filename)).convert('L').resize((px, px))) / 255
            if img is not None:
                images[count] = img.flatten()
                count += 1
    return images, count


images = np.zeros((15000, px * px), dtype='float64')
imagesY = np.zeros((15000, 15), dtype='int8')

countY = 0
count = 0
back_count = 0

# init train set
for folder in folders:
    path = 'Vegetable Images/train/' + folder
    images, count = load_images_from_folder(images, path, count)
    imagesY[back_count: count, countY:countY + 1] = 1
    countY += 1
    back_count = count
images, imagesY = shuffle(images, imagesY, random_state=0)

countY = 0
count = 0
back_count = 0

# init validation set
images_validation = np.zeros((3000, px * px), dtype='float64')
images_validation_Y = np.zeros((3000, 15), dtype='int8')
for folder in folders:
    path = 'Vegetable Images/validation/' + folder
    images_validation, count = load_images_from_folder(images_validation, path, count)
    images_validation_Y[back_count: count, countY:countY + 1] = 1
    countY += 1
    back_count = count


# sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    return np.exp(x) / np.exp(x).sum()


# tanh
def tanh(x):
    return np.tanh(x)


def derivative_tanh(x):
    return 1 - np.square(tanh(x))


# ReLU
def relu(x):
    x[x <= 0] = 0
    return x


def derivative_relu(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


# initialize weight
def initialize_weight(layers):
    w = dict()
    b = dict()
    for i in range(len(layers) - 1):
        w[i] = np.random.uniform(-0.3, 0.3, (layers[i], layers[i + 1]))
        b[i] = np.zeros(layers[i + 1])
    return w, b


def back_prop(bias, weight, activation, y_batch, function_name, batch, alpha, z, gradient_array):
    if function_name == "sigmoid":
        index1 = list(bias.keys())[-1]
        index2 = index1
        dBias = (activation[index1 + 1] - y_batch) / batch
        # print(f"dbias {index1}", np.sum(dBias) - gradient_array[0][index1])
        bias[index1] = bias[index1] - alpha * np.sum(dBias, axis=0)

        while index1 > 0:
            index1 -= 1
            dBias = np.dot(dBias, weight[index1 + 1].T) * activation[index1 + 1] * (1 - activation[index1 + 1])
            # print(f"dbias {index1}", np.sum(dBias) - gradient_array[0][index1])
            bias[index1] = bias[index1] - alpha * np.sum(dBias, axis=0)

        weight_copy = weight.copy()
        cons = (activation[index2 + 1] - y_batch) / batch
        dweight = np.dot(activation[index2].T, cons)
        # print(f"dweight {index2}", np.sum(dweight) - gradient_array[1][index2])
        weight[index2] = weight[index2] - alpha * dweight
        while index2 > 0:
            index2 -= 1
            cons = np.dot(cons, weight_copy[index2 + 1].T) * activation[index2 + 1] * (1 - activation[index2 + 1])
            dweight = np.dot(activation[index2].T, cons)
            # print(f"dweight {index2}", np.sum(dweight) - gradient_array[1][index2])
            weight[index2] = weight[index2] - alpha * dweight


    elif function_name == "relu":
        index1 = list(bias.keys())[-1]
        index2 = index1
        dBias = (activation[index1 + 1] - y_batch) / batch
        bias[index1] = bias[index1] - alpha * np.sum(dBias, axis=0)

        while index1 > 0:
            index1 -= 1
            dBias = np.dot(dBias, weight[index1 + 1].T) * derivative_relu(z[index1])
            bias[index1] = bias[index1] - alpha * np.sum(dBias, axis=0)

        weight_copy = weight.copy()
        cons = (activation[index2 + 1] - y_batch) / batch
        dweight = np.dot(activation[index2].T, cons)
        # print(f"dweight {index2}", np.sum(dweight) - gradient_array[1][index2])
        weight[index2] = weight[index2] - alpha * dweight
        while index2 > 0:
            index2 -= 1
            cons = np.dot(cons, weight_copy[index2 + 1].T) * derivative_relu(z[index2])
            dweight = np.dot(activation[index2].T, cons)
            # print(f"dweight {index2}", np.sum(dweight) - gradient_array[1][index2])
            weight[index2] = weight[index2] - alpha * dweight
    else:
        index1 = list(bias.keys())[-1]
        index2 = index1
        dBias = (activation[index1 + 1] - y_batch) / batch
        bias[index1] = bias[index1] - alpha * np.sum(dBias, axis=0)

        while index1 > 0:
            index1 -= 1
            dBias = np.dot(dBias, weight[index1 + 1].T) * derivative_tanh(z[index1])
            bias[index1] = bias[index1] - alpha * np.sum(dBias, axis=0)

        weight_copy = weight.copy()
        cons = (activation[index2 + 1] - y_batch) / batch
        dweight = np.dot(activation[index2].T, cons)
        # print(f"dweight {index2}", np.sum(dweight) - gradient_array[1][index2])
        weight[index2] = weight[index2] - alpha * dweight
        while index2 > 0:
            index2 -= 1
            cons = np.dot(cons, weight_copy[index2 + 1].T) * derivative_tanh(z[index2])
            dweight = np.dot(activation[index2].T, cons)
            # print(f"dweight {index2}", np.sum(dweight) - gradient_array[1][index2])
            weight[index2] = weight[index2] - alpha * dweight
    return bias, weight


def forward_prog(layers, z, activation, weight, bias, func, hidden_numbers):
    k = 0
    for j in range(len(layers) - 1):
        z[j] = np.dot(activation[j], weight[j]) + bias[j]
        activation[j + 1] = func(z[j])
        k = j
    activation[k + 1] = sigmoid(z[k])
    return activation[hidden_numbers + 1], z, activation


def gradient_check(loss, hidden_numbers, weight, bias, batch, y_batch, layers, z, activation, func):
    result = np.zeros((2, hidden_numbers + 1), dtype='float64')
    for i in range(hidden_numbers + 1):
        temp = weight[i]
        weight[i] = weight[i] + epsilon
        y_hat = forward_prog(layers, z, activation, weight, bias, func, hidden_numbers)[0]
        loss1 = -1 / batch * (np.sum(y_batch * np.log(y_hat) + (1 - y_batch) * np.log(1 - y_hat)))
        result[1][i] = (loss1 - loss) / (epsilon * 2)
        weight[i] = temp
        temp = bias[i]
        bias[i] += epsilon
        y_hat = forward_prog(layers, z, activation, weight, bias, func, hidden_numbers)[0]
        loss1 = -1 / batch * (np.sum(y_batch * np.log(y_hat) + (1 - y_batch) * np.log(1 - y_hat)))
        result[0][i] = (loss1 - loss) / epsilon
        bias[i] = temp
    return result


def listToStr(liste):
    """
    make list to str
    :param liste: is a list
    :return: string
    """
    s = ""
    for i in liste:
        s += str(i) + " "
    return s


# initialize NN
def neural_network(func, layers, alpha, batch, hidden_numbers=0):
    """
    make NN according to given params
    :param func: such as sigmoid, relu, tanh
    :param layers: hidden layers
    :param alpha: learning rate
    :param batch: batch size
    :param hidden_numbers: number of hidden layers
    :return:
    """
    if hidden_numbers == 0:
        layers = []
    layers.insert(0, px * px)
    layers.append(15)
    weight, bias = initialize_weight(layers)
    z = dict()
    activation = dict()
    los_ = 99999

    while True:
        for e in range(10):
            counter = 0
            for i in range(0, 15000 - batch, batch):
                counter += batch
                activation[0] = images[i: i + batch]
                y_batch = imagesY[i: i + batch]

                # forward prog
                y_hat, z, activation = forward_prog(layers, z, activation, weight, bias, func, hidden_numbers)

                # loss function
                loss = -1 / batch * (
                    np.sum(y_batch * np.log(y_hat) + (1 - y_batch) * np.log(1 - y_hat)))

                # gradient_array = gradient_check(loss, hidden_numbers, weight, bias, batch, y_batch, layers, z, activation,
                # func)
                # back prog
                bias, weight = back_prop(bias, weight, activation, y_batch, func.__name__, batch, alpha, z,
                                         gradient_array=[])

        # Validation part
        activation[0] = images_validation
        y_batch = images_validation_Y
        y_hat, z, activation = forward_prog(layers, z, activation, weight, bias, func, hidden_numbers)
        new_loss = (-1 / batch * (
            np.sum(y_batch * np.log(y_hat) + (1 - y_batch) * np.log(1 - y_hat))))

        # calculate accuracy
        y_hat = (y_hat == y_hat.max(axis=1)[:, None]).astype(int)
        acc = accuracy_score(y_batch, y_hat)
        pre = precision_score(y_batch, y_hat, average=None, zero_division=0)
        recall = recall_score(y_batch, y_hat, average=None, zero_division=0)
        f1 = f1_score(y_batch, y_hat, average=None, zero_division=0)
        cnf = confusion_matrix(y_batch.argmax(axis=1), y_hat.argmax(axis=1))
        if new_loss >= los_:
            break
        los_ = new_loss
    return func.__name__, listToStr(
        layers), alpha, batch, hidden_numbers, acc, pre, recall, f1, cnf, weight, bias, layers, func


images = np.zeros((15000, px * px), dtype='float64')
imagesY = np.zeros((15000, 15), dtype='int8')

countY = 0
count = 0
back_count = 0

# init train set
for folder in folders:
    path = 'Vegetable Images/train/' + folder
    images, count = load_images_from_folder(images, path, count)
    imagesY[back_count: count, countY:countY + 1] = 1
    countY += 1
    back_count = count
images, imagesY = shuffle(images, imagesY, random_state=0)

countY = 0
count = 0
back_count = 0

# init validation set
images_validation = np.zeros((3000, px * px), dtype='float64')
images_validation_Y = np.zeros((3000, 15), dtype='int8')
for folder in folders:
    path = 'Vegetable Images/validation/' + folder
    images_validation, count = load_images_from_folder(images_validation, path, count)
    images_validation_Y[back_count: count, countY:countY + 1] = 1
    countY += 1
    back_count = count

# for loop makes the code very very slow (only one parameter takes more than 8 hours)
results = []
hidden1 = 25
hidden2 = 20
for i in range(81):
    results.append([])
results[0] = neural_network(sigmoid, [], 0.005, 16, 0)
results[1] = neural_network(sigmoid, [], 0.005, 32, 0)
results[2] = neural_network(sigmoid, [], 0.005, 64, 0)
results[3] = neural_network(sigmoid, [], 0.01, 16, 0)
results[4] = neural_network(sigmoid, [], 0.01, 32, 0)
results[5] = neural_network(sigmoid, [], 0.01, 64, 0)
results[6] = neural_network(sigmoid, [], 0.02, 16, 0)
results[7] = neural_network(sigmoid, [], 0.02, 32, 0)
results[8] = neural_network(sigmoid, [], 0.02, 64, 0)
results[9] = neural_network(sigmoid, [hidden1], 0.005, 16, 1)
results[10] = neural_network(sigmoid, [hidden1], 0.005, 32, 1)
results[11] = neural_network(sigmoid, [hidden1], 0.005, 64, 1)
results[12] = neural_network(sigmoid, [hidden1], 0.01, 16, 1)
results[13] = neural_network(sigmoid, [hidden1], 0.01, 32, 1)
results[14] = neural_network(sigmoid, [hidden1], 0.01, 64, 1)
results[15] = neural_network(sigmoid, [hidden1], 0.02, 16, 1)
results[16] = neural_network(sigmoid, [hidden1], 0.02, 32, 1)
results[17] = neural_network(sigmoid, [hidden1], 0.02, 64, 1)
results[18] = neural_network(sigmoid, [hidden1, hidden2], 0.005, 16, 2)
results[19] = neural_network(sigmoid, [hidden1, hidden2], 0.005, 32, 2)
results[20] = neural_network(sigmoid, [hidden1, hidden2], 0.005, 64, 2)
results[21] = neural_network(sigmoid, [hidden1, hidden2], 0.01, 16, 2)
results[22] = neural_network(sigmoid, [hidden1, hidden2], 0.01, 32, 2)
results[23] = neural_network(sigmoid, [hidden1, hidden2], 0.01, 64, 2)
results[24] = neural_network(sigmoid, [hidden1, hidden2], 0.02, 16, 2)
results[25] = neural_network(sigmoid, [hidden1, hidden2], 0.02, 32, 2)
results[26] = neural_network(sigmoid, [hidden1, hidden2], 0.02, 64, 2)

results[27] = neural_network(relu, [], 0.005, 16, 0)
results[28] = neural_network(relu, [], 0.005, 32, 0)
results[29] = neural_network(relu, [], 0.005, 64, 0)
results[30] = neural_network(relu, [], 0.01, 16, 0)
results[31] = neural_network(relu, [], 0.01, 32, 0)
results[32] = neural_network(relu, [], 0.01, 64, 0)
results[33] = neural_network(relu, [], 0.02, 16, 0)
results[34] = neural_network(relu, [], 0.02, 32, 0)
results[35] = neural_network(relu, [], 0.02, 64, 0)
results[36] = neural_network(relu, [hidden1], 0.005, 16, 1)
results[37] = neural_network(relu, [hidden1], 0.005, 32, 1)
results[38] = neural_network(relu, [hidden1], 0.005, 64, 1)
results[39] = neural_network(relu, [hidden1], 0.01, 16, 1)
results[40] = neural_network(relu, [hidden1], 0.01, 32, 1)
results[41] = neural_network(relu, [hidden1], 0.01, 64, 1)
results[42] = neural_network(relu, [hidden1], 0.02, 16, 1)
results[43] = neural_network(relu, [hidden1], 0.02, 32, 1)
results[44] = neural_network(relu, [hidden1], 0.02, 64, 1)
results[45] = neural_network(relu, [hidden1, hidden2], 0.005, 16, 2)
results[46] = neural_network(relu, [hidden1, hidden2], 0.005, 32, 2)
results[47] = neural_network(relu, [hidden1, hidden2], 0.005, 64, 2)
results[48] = neural_network(relu, [hidden1, hidden2], 0.01, 16, 2)
results[49] = neural_network(relu, [hidden1, hidden2], 0.01, 32, 2)
results[50] = neural_network(relu, [hidden1, hidden2], 0.01, 64, 2)
results[51] = neural_network(relu, [hidden1, hidden2], 0.02, 16, 2)
results[52] = neural_network(relu, [hidden1, hidden2], 0.02, 32, 2)
results[53] = neural_network(relu, [hidden1, hidden2], 0.02, 64, 2)

results[54] = neural_network(tanh, [], 0.005, 16, 0)
results[55] = neural_network(tanh, [], 0.005, 32, 0)
results[56] = neural_network(tanh, [], 0.005, 64, 0)
results[57] = neural_network(tanh, [], 0.01, 16, 0)
results[58] = neural_network(tanh, [], 0.01, 32, 0)
results[59] = neural_network(tanh, [], 0.01, 64, 0)
results[60] = neural_network(tanh, [], 0.02, 16, 0)
results[61] = neural_network(tanh, [], 0.02, 32, 0)
results[62] = neural_network(tanh, [], 0.02, 64, 0)
results[63] = neural_network(tanh, [hidden1], 0.005, 16, 1)
results[64] = neural_network(tanh, [hidden1], 0.005, 32, 1)
results[65] = neural_network(tanh, [hidden1], 0.005, 64, 1)
results[66] = neural_network(tanh, [hidden1], 0.01, 16, 1)
results[67] = neural_network(tanh, [hidden1], 0.01, 32, 1)
results[68] = neural_network(tanh, [hidden1], 0.01, 64, 1)
results[69] = neural_network(tanh, [hidden1], 0.02, 16, 1)
results[70] = neural_network(tanh, [hidden1], 0.02, 32, 1)
results[71] = neural_network(tanh, [hidden1], 0.02, 64, 1)
results[72] = neural_network(tanh, [hidden1, hidden2], 0.005, 16, 2)
results[73] = neural_network(tanh, [hidden1, hidden2], 0.005, 32, 2)
results[74] = neural_network(tanh, [hidden1, hidden2], 0.005, 64, 2)
results[75] = neural_network(tanh, [hidden1, hidden2], 0.01, 16, 2)
results[76] = neural_network(tanh, [hidden1, hidden2], 0.01, 32, 2)
results[77] = neural_network(tanh, [hidden1, hidden2], 0.01, 64, 2)
results[78] = neural_network(tanh, [hidden1, hidden2], 0.02, 16, 2)
results[79] = neural_network(tanh, [hidden1, hidden2], 0.02, 32, 2)
results[80] = neural_network(tanh, [hidden1, hidden2], 0.02, 64, 2)

result = np.array(results)
results = sorted(results, key=lambda l: l[5], reverse=True)
print(results)

layers = results[0][-2]
func = results[0][-1]
w = results[0][-4]
b = results[0][-3]
acc = results[0][5]
cnf = results[0][10]
btch = results[0][3]
layers = [x for x in layers if x != 3600 and x != 15]

countY = 0
count = 0
back_count = 0

# init test set
images_test = np.zeros((3000, px * px), dtype='float64')
images_test_Y = np.zeros((3000, 15), dtype='int8')
for folder in folders:
    path = 'Vegetable Images/test/' + folder
    images_test, count = load_images_from_folder(images_test, path, count)
    images_test_Y[back_count: count, countY:countY + 1] = 1
    countY += 1
    back_count = count

print(w)
print(b)
print(cnf)
print(layers)

layers.insert(0, px * px)
layers.append(15)
activation = dict()
z = dict()
weight = w.copy()
bias = b.copy()
activation[0] = images_test
y_batch = images_test_Y
y_hat, z, activation = forward_prog(layers, z, activation, weight, bias, func, len(layers) - 2)
new_loss = (-1 / btch * (np.sum(y_batch * np.log(y_hat) + (1 - y_batch) * np.log(1 - y_hat))))
count = 0
for i in range(3000):
    if np.argmax(images_test_Y[i]) == np.argmax(y_hat[i]):
        count += 1
acc_test = count / 3000
# test accuracy
print(acc_test)

# test loss
print(new_loss)

# Visualize
for i in weight.keys():
    weight[i] = weight[i] - np.min(weight[i])
    weight[i] = weight[i] / np.max(weight[i])

    img = Image.fromarray(np.uint8(weight[i] * 255), 'L').resize((px, px))
    img.show()

# other relevent to report so we don't add here
