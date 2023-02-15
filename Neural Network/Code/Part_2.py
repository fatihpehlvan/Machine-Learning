from PIL import Image
import os
import numpy as np
import random
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import time
import copy
import sys
import torchmetrics

num_epochs=10
batchs_size = 32


#Loading data
data_dir = "Vegetable Images"
data_transforms = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor()])

trainloader = torch.utils.data.DataLoader(datasets.ImageFolder(os.path.join(data_dir,'train'), data_transforms), batch_size=batchs_size, shuffle=True)
validloader = torch.utils.data.DataLoader(datasets.ImageFolder(os.path.join(data_dir,'validation'), data_transforms), batch_size=batchs_size, shuffle=True)
testloader = torch.utils.data.DataLoader(datasets.ImageFolder(os.path.join(data_dir,'test'), data_transforms), batch_size=batchs_size, shuffle=True)
print("Initializing Dataloaders...")


# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(device)

model_1 = models.vgg19(weights='VGG19_Weights.DEFAULT')
model_1.classifier[6] = nn.Linear(4096,15)
model_1 = model_1.to(device)


def train_model(model, trainloader, validloader, criterion, optimizer, num_epochs=10):
    since = time.time()

    val_acc_history = []
    train_acc_history = []
    val_loss_history = []
    train_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    min_loss = sys.float_info.max

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 20)

        # Train part
        model.train()

        train_loss = 0.0
        train_corrects = 0

        # get the inputs; trainloader is a list of [inputs, labels]
        for inputs, labels in trainloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                # forward
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # The outputs are energies for the 10 classes.
                # The higher the energy for a class, the more the network thinks that the image is of the particular class.
                # So, let’s get the index of the highest energy:
                _, preds = torch.max(outputs, 1)

                # backward + optimize
                loss.backward()
                optimizer.step()

            # print statistics
            train_loss += loss.item() * inputs.size(0)
            train_corrects += (preds == labels).sum().item()

        epoch_loss = train_loss / len(trainloader.dataset)
        epoch_acc = train_corrects / len(trainloader.dataset)
        train_acc_history.append(epoch_acc)
        train_loss_history.append(epoch_loss)
        print('Train: Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        # Validation part
        model.eval()

        validation_loss = 0.0
        validation_corrects = 0

        # get the inputs; trainloader is a list of [inputs, labels]
        for inputs, labels in validloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            # since we're not training, we don't need to calculate the gradients for our outputs
            with torch.set_grad_enabled(False):
                # forward
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

            # print statistics
            validation_loss += loss.item() * inputs.size(0)
            validation_corrects += (preds == labels).sum().item()

        epoch_loss = validation_loss / len(validloader.dataset)
        epoch_acc = validation_corrects / len(validloader.dataset)
        val_acc_history.append(epoch_acc)
        val_loss_history.append(epoch_loss)

        print('Validation: Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        if epoch_loss < min_loss:
            min_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Min Validation Loss: {:4f}'.format(min_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_acc_history, val_acc_history, train_loss_history, val_loss_history


def test_model(testloader, model):
    total_accuracy = 0.0
    total_recall = 0.0
    total_precision = 0.0
    total_f1 = 0.0

    accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=15).to(device)
    recall = torchmetrics.Recall(task="multiclass", average='macro', num_classes=15).to(device)
    precision = torchmetrics.Precision(task="multiclass", average='macro', num_classes=15).to(device)
    f1 = torchmetrics.F1Score(task="multiclass", average='macro', num_classes=15).to(device)

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            total_accuracy += (accuracy(preds, labels).item() * inputs.size(0))
            total_recall += (recall(preds, labels).item() * inputs.size(0))
            total_precision += (precision(preds, labels).item() * inputs.size(0))
            total_f1 += (f1(preds, labels).item() * inputs.size(0))

    print('Accuracy: ', (total_accuracy / len(testloader.dataset)))
    print('Recall: ', (total_recall / len(testloader.dataset)))
    print('Precision: ', (total_precision / len(testloader.dataset)))
    print('F1:', (total_f1 / len(testloader.dataset)))

# Train and evaluate
model_1, train_acc_history, val_acc_history, train_loss_history, val_loss_history = train_model(model_1, trainloader, validloader, criterion, optimizer_1, num_epochs)
torch.save(model_1.state_dict(), 'model1')

plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.plot(range(1,num_epochs+1),val_loss_history,label="Validation loss")
plt.plot(range(1,num_epochs+1),train_loss_history,label="Train Loss")
plt.xticks(np.arange(1, num_epochs+1, 1.0))
plt.xlabel("Training Epochs")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(range(1,num_epochs+1),val_acc_history,label="Validation Accuracy")
plt.plot(range(1,num_epochs+1),train_acc_history,label="Train Accuracy")
plt.xlabel("Training Epochs")
plt.ylabel("Accuracy")
plt.xticks(np.arange(1, num_epochs+1, 1.0))
plt.legend()

plt.show()

#Test
test_model(testloader, model_1)

#Part2.2
model_2 = models.vgg19(weights='VGG19_Weights.DEFAULT')
model_2.classifier[6] = nn.Linear(4096,15)
model_2 = model_2.to(device)

# requires_grad of all parameters except FC1 and FC2 must be False.
for name, param in model_2.named_parameters():
    if ("classifier.0" in name or "classifier.3" in name):
        param.requires_grad=True
    else:
        param.requires_grad=False
    print(name, param.requires_grad)

# Observe that all parameters are being optimized
optimizer_2 = optim.SGD(model_2.parameters(), lr=0.001, momentum=0.9)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# Train and evaluate
model_2, train_acc_history, val_acc_history, train_loss_history, val_loss_history = train_model(model_2, trainloader, validloader, criterion, optimizer_2, num_epochs)
torch.save(model_1.state_dict(), 'model2')

plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.plot(range(1,num_epochs+1),val_loss_history,label="Validation loss")
plt.plot(range(1,num_epochs+1),train_loss_history,label="Train Loss")
plt.xticks(np.arange(1, num_epochs+1, 1.0))
plt.xlabel("Training Epochs")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(range(1,num_epochs+1),val_acc_history,label="Validation Accuracy")
plt.plot(range(1,num_epochs+1),train_acc_history,label="Train Accuracy")
plt.xlabel("Training Epochs")
plt.ylabel("Accuracy")

plt.xticks(np.arange(1, num_epochs+1, 1.0))
plt.legend()

plt.show()

test_model(testloader, model_2)
