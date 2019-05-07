from keras.preprocessing.image import ImageDataGenerator

import os
import sys
from keras.models import Model
from keras.datasets import cifar10
import keras
from collections import Counter
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from seaborn import countplot
plt.rcdefaults()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.model_buider import ModelBuilder

title = None
xlabel = None
ylabel = None

def samples_visuale(data, train_data=True):
    y_pos = np.arange(len(data['classes']))
    countImg = Counter(data['y_train']).values()
    if(train_data == False):
        countImg = Counter(data['y_test']).values()

    plt.bar(y_pos, countImg, align='center', alpha=0.5, color='blue')
    plt.xticks(y_pos, data['classes'])
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    plt.show()

def img_visuale(data):

    columns = 10
    rows = 3
    fig = plt.figure(figsize=[5, 5])
    for i in range(columns*rows):
        fig.add_subplot(rows, columns, i+1)
        curr_img = data['x_train'][i]
        curr_lbl = data['y_train'][i]
        plt.imshow(curr_img)
        plt.title(str(data['classes'][curr_lbl]))
    plt.show()

def loss_visuale(history,loss=True):
    yaxis = history['loss']
    val_yaxis = history.history['val_loss']
    if loss == False:
        yaxis = history['acc']
        val_yaxis = history['val_acc']
        
    epochs = range(1,len(yaxis)+1)

    plt.plot(epochs, yaxis, color='red', label='Training')
    plt.plot(epochs, val_yaxis, color='green', label='Validation')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()

################################

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
data = {
    'classes': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
    'x_train': x_train*1.0/255,
    'y_train': np.reshape(y_train, y_train.shape[0]),
    'x_test': x_test,
    'y_test': np.reshape(y_test, y_test.shape[0])
}



x_train = data['x_train']
y_train = data['y_train']
x_test = data['x_test']
y_test = data['y_test']
classes = data['classes']


# label encoding
num_class = len(classes)
y_train = to_categorical(y_train, num_class)
print(y_train.shape)
print(y_train[:5])

# split training data into training data and validation data
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train,test_size=0.1,random_state=42)


# Defining cnn model
input_shape = (32, 32,3)
model = ModelBuilder._custom_network(input_shape, num_class)
model.summary()

# data augmentation
datagen_train = ImageDataGenerator(rotation_range=90)
datagen_train.fit(x_train)
# for x_batch, y_batch in datagen_train.flow(x_train, y_train, batch_size=9):
#         # create a grid of 3x3 images
#     for i in range(0, 9):
#         plt.subplot(330 + 1 + i)
#         plt.imshow(x_batch[i])
#     # show the plot
#     plt.show()
#     break


#optimizer

# trainning 
history = model.fit(x_train,y_train,batch_size= 32,epochs=5,validation_data=(x_valid,y_valid),verbose = 1)
# history = model.fit_generator(datagen_train.flow(x_train, y_train, batch_size=32), validation_data=(x_valid, y_valid), epochs=5,steps_per_epoch=x_train.shape[0], verbose=1)

#  plotting training and validation loss or acc
loss_visuale(history.history)
loss_visuale(history.history,loss=False)

# Displaying original Image
img_visuale(data)

# Visualize CNN Layers
# layer_outputs = [layer.output for layer in model.layers]
# activation_model = Model(inputs=model.input, outputs=layer_outputs)
# activations = activation_model.predict(x_train[10].reshape(1,32,32,3))
 
# def display_activation(activations, col_size, row_size, act_index): 
#     activation = activations[act_index]
#     activation_index=0
#     fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
#     for row in range(0,row_size):
#         for col in range(0,col_size):
#             ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
#             activation_index += 1

