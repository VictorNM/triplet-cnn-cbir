import keras
from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from keras.models import Model, Sequential

FC1 = 'fc1'
FC2 = 'fc2'
FC3 = 'fc3'


def build_cnn_extractor(cnn_classifier, features_layer=FC2):
    cnn_extractor = Model(
        inputs=cnn_classifier.input,
        outputs=cnn_classifier.get_layer(features_layer).output)

    return cnn_extractor
