import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import itertools

from data import load_datasets
from model import MaskNet


device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint = torch.load(
    r"checkpoints/lr1e-5/masknet-epoch=37-val_loss=0.04.ckpt", map_location=device,
)


state_dict = checkpoint["state_dict"]
state_dict = {k.partition("_model.")[2]: v for k, v in state_dict.items()}

model = MaskNet()
model.load_state_dict(state_dict)
model.eval()

train_dataset, val_dataset, test_dataset = load_datasets(r"dataset")
testloader = DataLoader(test_dataset, batch_size=64)


correct = 0
total = 0
prediction_labels = []
true_labels = []

with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        true_labels += labels.cpu().numpy().tolist()
        prediction_labels += predicted.cpu().numpy().tolist()


print("Accuracy of the network on the test images: %d %%" % (100 * correct / total))
Y_test = np.array(true_labels)
y_score = np.array(prediction_labels)
target_names = test_dataset.classes
print(classification_report(Y_test, y_score, target_names=target_names))


# Plotting the confusion matrix
def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
):

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


# Compute confusion matrix
cnf_matrix = confusion_matrix(Y_test, y_score)
print("Confusion matrix :")
print(cnf_matrix)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(
    cnf_matrix, classes=test_dataset.classes, normalize=False, title="Confusion Matrix"
)
plt.savefig('cm.png')
plt.show()

********************************************************************************************************

device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint = torch.load(
    r"/content/drive/MyDrive/comp6721_project1-main/checkpoints/kfold/masknet-epoch=57-val_accuracy=0.989.ckpt", map_location=device,
)


state_dict = checkpoint["state_dict"]
state_dict = {k.partition("_model.")[2]: v for k, v in state_dict.items()}

model = MaskNet()
model.load_state_dict(state_dict)
model.eval()

train_load, test_load = load_datasets_v2(r'/content/drive/MyDrive/comp6721_project1-main/dataset/v2')
testloader = DataLoader(test_load, batch_size=64)


correct = 0
total = 0
prediction_labels = []
true_labels = []

with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        true_labels += labels.cpu().numpy().tolist()
        prediction_labels += predicted.cpu().numpy().tolist()


print("Accuracy of the network on the test images: %d %%" % (100 * correct / total))
Y_test = np.array(true_labels)
y_score = np.array(prediction_labels)
target_names = test_load.classes
print(classification_report(Y_test, y_score, target_names=target_names))


# Plotting the confusion matrix
def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
):

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


# Compute confusion matrix
cnf_matrix = confusion_matrix(Y_test, y_score)
print("Confusion matrix :")
print(cnf_matrix)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(
    cnf_matrix, classes=test_load.classes, normalize=False, title="Confusion Matrix"
)
plt.savefig('cm.png')
plt.show()

