import keras
from keras.applications import vgg16
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten, Dropout, Activation
from .model import Model


class ModelBuilder(object):
    @staticmethod
    def load(model_config, input_shape, num_classes):
        model_name = model_config.name
        if model_name == 'custom':
            return ModelBuilder._custom_model(input_shape, num_classes)

        raise ValueError("config invalid")

    @staticmethod
    def _custom_model(input_shape, num_classes):
        cnn = Sequential()
        cnn.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
        cnn.add(Activation('relu'))
        cnn.add(Conv2D(32, (3, 3)))
        cnn.add(Activation('relu'))
        cnn.add(MaxPooling2D(pool_size=(2, 2)))
        cnn.add(Dropout(0.25))

        cnn.add(Conv2D(64, (3, 3), padding='same'))
        cnn.add(Activation('relu'))
        cnn.add(Conv2D(64, (3, 3)))
        cnn.add(Activation('relu'))
        cnn.add(MaxPooling2D(pool_size=(2, 2)))
        cnn.add(Dropout(0.25))

        cnn.add(Flatten())
        cnn.add(Dense(512))
        cnn.add(Activation('relu'))
        cnn.add(Dropout(0.5))
        cnn.add(Dense(num_classes))
        cnn.add(Activation('softmax'))

        # initiate RMSprop optimizer
        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

        # Let's train the cnn using RMSprop
        cnn.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        model = Model()
        model.cnn_classifier = cnn
        model.cnn_feature_layer_name = 'dropout_3'

        return model
