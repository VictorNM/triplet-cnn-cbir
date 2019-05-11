import keras
from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten, Dropout


CUSTOM = 'custom'
VGG16 = 'vgg16'
FC1 = 'fc1'
FC2 = 'fc2'
FC3 = 'fc3'


def build_cnn_classifier(model_name, input_shape, num_classes):
    if model_name == 'custom':
        print('Model custom has been built')
        return _custom_model(input_shape, num_classes)
    if model_name == 'vgg16':
        print('Model VGG16 has been built')
        return _vgg16(input_shape, num_classes)

    raise ValueError("Model not implemented")


def build_cnn_extractor(cnn_classifier, features_layer=FC2):
    cnn_extractor = Model(
        inputs=cnn_classifier.input,
        outputs=cnn_classifier.get_layer(features_layer).output)

    return cnn_extractor


# Pattern; INPUT -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K -> FC
# N <= 3, M >= 0, K <= 3
def _custom_model(input_shape, num_classes):
    cnn = Sequential()

    cnn.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    cnn.add(Conv2D(64, (3, 3), activation='relu'))
    cnn.add(MaxPooling2D())

    cnn.add(Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    cnn.add(Conv2D(128, (3, 3), activation='relu'))
    cnn.add(MaxPooling2D())

    cnn.add(Conv2D(256, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    cnn.add(Conv2D(256, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    cnn.add(Conv2D(256, (3, 3), activation='relu'))
    cnn.add(MaxPooling2D())

    cnn.add(Conv2D(512, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    cnn.add(Conv2D(512, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    cnn.add(Conv2D(512, (3, 3), activation='relu'))
    cnn.add(MaxPooling2D())

    cnn.add(Flatten())
    cnn.add(Dense(2048, activation='relu', name=FC1))
    cnn.add(Dense(2048, activation='relu', name=FC2))
    cnn.add(Dense(num_classes, activation='softmax'))

    return cnn


def _vgg16(input_shape, num_classes):
    cnn = Sequential()

    cnn.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    cnn.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    cnn.add(MaxPooling2D((2, 2), strides=(2, 2)))

    cnn.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    cnn.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    cnn.add(MaxPooling2D((2, 2), strides=(2, 2)))

    cnn.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    cnn.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    cnn.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    cnn.add(MaxPooling2D((2, 2), strides=(2, 2)))

    cnn.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    cnn.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    cnn.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    cnn.add(MaxPooling2D((2, 2), strides=(2, 2)))

    cnn.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    cnn.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    cnn.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    cnn.add(MaxPooling2D((2, 2), strides=(2, 2)))

    cnn.add(Flatten())
    cnn.add(Dense(4096, activation='relu', name=FC1))
    cnn.add(Dense(4096, activation='relu', name=FC2))
    cnn.add(Dense(num_classes, activation='softmax', name='predictions'))

    return cnn
