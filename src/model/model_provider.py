import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten, Dropout, Activation


class ModelProvider(object):

    EXTRACT_LAYER_NAME = 'extract_layer'

    def __init__(self, model_config):
        self._config = model_config

    def build_cnn_classifier(self, input_shape, num_classes):
        model_name = self._config.name
        if model_name == 'custom':
            return self._custom_model(input_shape, num_classes)

        raise ValueError("Model not implemented")

    def build_cnn_extractor(self, cnn_classifier):
        cnn_extractor = keras.models.Model(
            inputs=cnn_classifier.input,
            outputs=cnn_classifier.get_layer(ModelProvider.EXTRACT_LAYER_NAME).output)

        return cnn_extractor

    def build_deep_ranking_extractor(self, cnn_extractor):
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

    def _custom_model(self, input_shape, num_classes):
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
        cnn.add(Dense(512, name=ModelProvider.EXTRACT_LAYER_NAME))
        cnn.add(Activation('relu'))
        cnn.add(Dropout(0.5))
        cnn.add(Dense(num_classes))
        cnn.add(Activation('softmax'))

        opt = keras.optimizers.SGD(lr=self._config.learning_rate)

        cnn.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        return cnn
