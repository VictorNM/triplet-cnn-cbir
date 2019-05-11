from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2


def normalize(dataset, input_shape):
    return {
        **dataset,
        'x_train': _normalize(dataset['x_train'], input_shape),
        'x_valid': _normalize(dataset['x_valid'], input_shape),
        'x_test': _normalize(dataset['x_test'], input_shape)
    }


def _normalize(x, input_shape):
    input_size = input_shape[:-1]
    n = x.shape[0]
    new_x = np.ndarray(shape=((n, ) + input_shape))
    for i in range(n):
        im = x[i]
        new_x[i] = cv2.resize(im, input_size)

    return new_x / 255.0


def augment(dataset, augment_params):
    print('Augmenting dataset...')
    x_train, y_train = dataset['x_train'], dataset['y_train']
    new_x_train, new_y_train = _augment(x_train, y_train, augment_params)
    new_dataset = {
        **dataset,
        'x_train': new_x_train,
        'y_train': new_y_train
    }

    print('Augmented')

    return new_dataset


def _augment(x, y, augment_params):
    generator = ImageDataGenerator(**augment_params).flow(x, y)
    return _convert_generator_to_data(generator)


def _convert_generator_to_data(generator):
    x = np.ndarray(generator.x.shape, dtype=generator.dtype)
    y = np.ndarray(shape=generator.n, dtype=generator.dtype)
    for i in range(generator.__len__()):
        x_batch, y_batch = generator.next()
        start_index = i * generator.batch_size
        end_index = (i + 1) * generator.batch_size
        x[start_index:end_index] = x_batch
        y[start_index:end_index] = y_batch

    return x, y
