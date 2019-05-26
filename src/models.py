import keras.backend as K
from keras.layers import (Conv2D, Dense, Dropout, Flatten, Input, Layer,
                          MaxPooling2D)
from keras.models import Model, Sequential

FC1 = 'fc1'
FC2 = 'fc2'
FC3 = 'fc3'


class TripletLossLayer(Layer):
    def __init__(self, margin, **kwargs):
        self.margin = margin
        super(TripletLossLayer, self).__init__(**kwargs)
    
    def triplet_loss(self, inputs):
        a, p, n = inputs
        a = K.l2_normalize(a)
        p = K.l2_normalize(p)
        n = K.l2_normalize(n)
        p_dist = K.sqrt(K.sum(K.square(a-p), axis=-1))
        n_dist = K.sqrt(K.sum(K.square(a-n), axis=-1))
        return K.mean(K.maximum(p_dist - n_dist + self.margin, 0), axis=0)
    
    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss


def build_cnn_extractor(cnn_classifier, features_layer=FC2):
    cnn_extractor = Model(
        inputs=cnn_classifier.input,
        outputs=cnn_classifier.get_layer(features_layer).output)

    return cnn_extractor


def build_triplet_extractor(cnn_classifier, margin=0.2, features_layer=FC2):
    extractor = Model(
        inputs=cnn_classifier.input,
        outputs=cnn_classifier.get_layer(features_layer).output)
        
    # freeze all layer ixcept two last FC layers
    for i in range(len(extractor.layers)):
        if extractor.layers[i].name != 'fc2' and extractor.layers[i].name != 'fc1':
            extractor.layers[i].trainable = False

    extractor.name = 'extractor'
    input_shape = cnn_classifier.layers[0].input_shape[1:]

    a_in = Input(shape=input_shape)
    p_in = Input(shape=input_shape)
    n_in = Input(shape=input_shape)

    a_cnn = extractor(a_in)
    p_cnn = extractor(p_in)
    n_cnn = extractor(n_in)

    triplet_loss_layer = TripletLossLayer(margin=margin, name='triplet_loss')([a_cnn, p_cnn, n_cnn])

    return Model(inputs=[a_in, p_in, n_in], outputs=triplet_loss_layer)


def vgg16(input_shape, num_classes):
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
    cnn.add(Dense(4096, activation='relu', name='fc1'))
    cnn.add(Dense(4096, activation='relu', name='fc2'))
    cnn.add(Dense(num_classes, activation='softmax', name='predictions'))

    return cnn


def vgg13(input_shape, num_classes):
    cnn = Sequential()

    cnn.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    cnn.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    cnn.add(MaxPooling2D((2, 2), strides=(2, 2)))

    cnn.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    cnn.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    cnn.add(MaxPooling2D((2, 2), strides=(2, 2)))

    cnn.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    cnn.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    cnn.add(MaxPooling2D((2, 2), strides=(2, 2)))

    cnn.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    cnn.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    cnn.add(MaxPooling2D((2, 2), strides=(2, 2)))

    cnn.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    cnn.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    cnn.add(MaxPooling2D((2, 2), strides=(2, 2)))

    cnn.add(Flatten())
    cnn.add(Dense(4096, activation='relu', name='fc1'))
    cnn.add(Dense(4096, activation='relu', name='fc2'))
    cnn.add(Dense(num_classes, activation='softmax', name='predictions'))

    return cnn


def vgg11(input_shape, num_classes):
    cnn = Sequential()

    cnn.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    cnn.add(MaxPooling2D((2, 2), strides=(2, 2)))

    cnn.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    cnn.add(MaxPooling2D((2, 2), strides=(2, 2)))

    cnn.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    cnn.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    cnn.add(MaxPooling2D((2, 2), strides=(2, 2)))

    cnn.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    cnn.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    cnn.add(MaxPooling2D((2, 2), strides=(2, 2)))

    cnn.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    cnn.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    cnn.add(MaxPooling2D((2, 2), strides=(2, 2)))

    cnn.add(Flatten())
    cnn.add(Dense(4096, activation='relu', name='fc1'))
    cnn.add(Dense(4096, activation='relu', name='fc2'))
    cnn.add(Dense(num_classes, activation='softmax', name='predictions'))

    return cnn
