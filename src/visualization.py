import os
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

plt.rcdefaults()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

title = None
xlabel = None
ylabel = None


def show_class_balance(data, train_data=True):
    y_pos = np.arange(len(data['classes']))
    countImg = Counter(data['y_train']).values()
    if (train_data == False):
        countImg = Counter(data['y_test']).values()

    plt.bar(y_pos, countImg, align='center', alpha=0.5, color='blue')
    plt.xticks(y_pos, data['classes'])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.show()


def visualize_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Valid'], loc='upper left')
    plt.show()


def visualize_acc(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Valid'], loc='upper left')
    plt.show()


def show_single_image(image):
    plt.imshow(image)


def show_many_images(images, row=4, col=4):
    f, axarr = plt.subplots(row, col)
    for i in range(row):
        for j in range(col):
            axarr[i, j].imshow(images[i * row + j])
