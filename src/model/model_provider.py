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

    # Pattern; INPUT -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K -> FC
    # N <= 3, M >= 0, K <= 3
    def _custom_model(self, input_shape, num_classes):
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
        cnn.add(Dense(512, activation='relu'))
        cnn.add(Dropout(0.5))
        cnn.add(Dense(512, activation='relu', name=ModelProvider.EXTRACT_LAYER_NAME))
        cnn.add(Dropout(0.5))
        cnn.add(Dense(num_classes, activation='softmax'))

        return cnn