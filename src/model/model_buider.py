import keras
from keras.applications import vgg16
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten, Dropout, Activation


class ModelBuilder(object):
    @staticmethod
    def load(model_config, input_shape, num_classes):
        model_name = model_config.name
        if model_name == 'custom':
            return ModelBuilder._custom_network(input_shape, num_classes)
        if model_name == 'vgg16':
            return ModelBuilder._vgg16()

        raise ValueError("config invalid")

    @staticmethod
    def _custom_network(input_shape, num_classes):
        model = Sequential()

        model.add(Conv2D(32, (3, 3),name='layer_conv1',padding='same',input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2),name='maxPool1'))
        
        model.add(Conv2D(32, (3, 3),name='layer_conv2',padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2),name='maxPool2'))

        model.add(Conv2D(64, (3, 3),name='conv3', padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2),name='maxPool3'))

        model.add(Flatten())
        model.add(Dense(512,name='fc0'))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes,name='fc1'))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Activation('softmax'))

        # initiate RMSprop optimizer
        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

        # Let's train the model using RMSprop
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])

        return model

    @staticmethod
    def _vgg16():
        return vgg16.VGG16(weight=None, input_shape=(32, 32, 3), classes=10)

