import keras
from keras.applications import vgg16
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten, Dropout, Activation
from .model import Model


class Models(object):

    EXTRACT_LAYER_NAME = 'extract_layer'

    @staticmethod
    def make_cnn_classifier(model_config, input_shape, num_classes):
        model_name = model_config.name
        if model_name == 'custom':
            return Models._custom_model(input_shape, num_classes)

        raise ValueError("Invalid configuration")

    @staticmethod
    def make_cnn_extractor(cnn_classifier):
        cnn_extractor = keras.models.Model(
            inputs=cnn_classifier.input,
            outputs=cnn_classifier.get_layer(Models.EXTRACT_LAYER_NAME).output)

        return cnn_extractor

    @staticmethod
    def make_deep_ranking_extractor(cnn_extractor):
        # freeze CNN's layers
        for layer in cnn_extractor.layers:
            layer.trainable = False

        x = cnn_extractor.output
        feature_layer = keras.layers.Dense(32)(x)
        deep_ranking_extractor = keras.models.Model(
            inputs=cnn_extractor.input,
            outputs=feature_layer
        )

        return deep_ranking_extractor

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
        cnn.add(Dense(512, name=Models.EXTRACT_LAYER_NAME))
        cnn.add(Activation('relu'))
        cnn.add(Dropout(0.5))
        cnn.add(Dense(num_classes))
        cnn.add(Activation('softmax'))

        # initiate RMSprop optimizer
        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

        # Let's train the cnn using RMSprop
        cnn.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        return cnn
