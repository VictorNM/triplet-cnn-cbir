from keras.models import Model
from keras.datasets import cifar10
import keras
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
plt.rcdefaults()


class BarChar:
    def __init__(self):
        self.title = None
        self.xlabel = None
        self.ylabel = None

    def plot_bar_char(self, dataset, train_data=True):
        y_pos = np.arange(len(dataset['classes']))
        countImg = ()
        if(train_data == True):
            countImg = Counter(dataset['y_train']).values()
        else:
            countImg = Counter(dataset['y_test']).values()
        plt.bar(y_pos, countImg, align='center', alpha=0.5, color='blue')
        plt.xticks(y_pos, dataset['classes'])
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.title)
        plt.show()

    def display_img(self, dataset):
        columns = 10
        rows = 3

        fig = plt.figure(figsize=[5, 5])
        for i in range(columns*rows):
            fig.add_subplot(rows, columns, i+1)
            curr_img = dataset['x_train'][i]
            curr_lbl = dataset['y_train'][i]
            plt.imshow(curr_img)
            plt.title(str(dataset['classes'][curr_lbl]))
        plt.show()

    def draw_loss_accurancy(self, train_x, test_x):
        # train_loss = []
        # test_loss = []
        # train_accuracy = []
        # test_accuracy = []

        plt.plot(range(len(train_x)), train_x,
                 color='blue', label='Training x')
        plt.plot(range(len(train_x)), test_x, color='red', label='Test x')
        plt.title('Training and Test x')
        plt.xlabel('Epochs ', fontsize=16)
        plt.ylabel('Loss', fontsize=16)
        plt.legend()
        plt.figure()

        plt.show()


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
data = {
    'classes': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
    'x_train': x_train,
    'y_train': np.reshape(y_train, y_train.shape[0]),
    'x_test': x_test,
    'y_test': np.reshape(y_test, y_test.shape[0])
}

# Desplaying above image after layer
# layer_outputs = [layer.output for layer in model.layers]
# activation_model = Model(inputs=model.input, outputs=layer_outputs)
# activations = activation_model.predict(X_train[10].reshape(1, 28, 28, 1))


# def display_activation(activations, col_size, row_size, act_index):
#     activation = activations[act_index]
#     activation_index = 0
#     fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5, col_size*1.5))
#     for row in range(0, row_size):
#         for col in range(0, col_size):
#             ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
#             activation_index += 1

# test = BarChar()
# test.display_img(data)
# test.title = 'Test'
# test.ylabel = 'test'
# test.plot_bar_char(data,False)
