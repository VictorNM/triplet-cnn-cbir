import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2


class DataProcessor:
    def __init__(self, data_config):
        self._config = data_config

    def normalize(self, x):
        input_shape = self._config.input_shape
        input_size = input_shape[:-1]
        n = x.shape[0]
        new_x = np.ndarray(shape=((n, ) + input_shape))
        for i in range(n):
            im = x[i]
            new_x[i] = cv2.resize(im, input_size)

        return new_x / 255.0

    def augment(self, x, y):
        data_augmentation = self._config.data_augmentation
        datagen = ImageDataGenerator(**data_augmentation)
        generator = datagen.flow(x, y)
        x, y = self._convert_generator_to_data(generator)
        return x, y

    def _convert_generator_to_data(self, generator):
        x = np.ndarray(generator.x.shape, dtype=generator.dtype)
        y = np.ndarray(shape=generator.n, dtype=generator.dtype)
        for i in range(generator.__len__()):
            x_batch, y_batch = generator.next()
            print(y_batch.shape)
            start_index = i * generator.batch_size
            end_index = (i + 1) * generator.batch_size
            x[start_index:end_index] = x_batch
            y[start_index:end_index] = y_batch

        return x, y