import os
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist, cifar10
import numpy as np
from six.moves import cPickle as pickle


class DataProvider:
    def __init__(self, data_root):
        self._data_root = data_root

    def load(self, dataset_name, dataset_type='raw', train_samples=0, test_samples=0):
        """
        :return: a dictionary include
            - 'classes': name of classes in the dataset
                    type list of string
            - 'x_train', 'x_test': data for training and testing,
                    shape=(num_samples, num_rows, num_col, num_channels)
                    type uint8
                    range [0..255]
            - 'y_train', 'y_test': label for training and testing
                    shape=(num_samples, )
                    type uint8
                    range [0...num_classes]
        """

        dataset = None
        try:
            print('Try to read from pickle file...')
            dataset = self._load_from_pickle_file(dataset_name, dataset_type)
        except FileNotFoundError:
            print('Pickle file not found, try to read from directory')
            if dataset_name == 'cifar-10':
                dataset = self._load_cifar10()
            elif dataset_name == 'vehicles':
                dataset = self._load_vehicles(dataset_type)
            elif dataset_name == 'standford_online_products':
                dataset = self._load_standford_online_product(dataset_type)
            else:
                raise NotImplementedError

            # save pickle file for later
            self._save_pickle(dataset, dataset_name, dataset_type)

        if train_samples > 0:
            dataset['x_train'] = dataset['x_train'][:train_samples]
            dataset['y_train'] = dataset['y_train'][:train_samples]

        if test_samples > 0:
            dataset['x_test'] = dataset['x_test'][:test_samples]
            dataset['y_test'] = dataset['y_test'][:test_samples]

        return dataset

    def _load_from_pickle_file(self, dataset_name, dataset_type):
        pickle_file_name = dataset_name + '.pickle'
        pickle_file_path = os.path.join(self._data_root, dataset_type, pickle_file_name)
        if os.path.exists(pickle_file_path):
            with open(pickle_file_path, 'rb') as f:
                dataset = pickle.load(f)
                return dataset

        raise FileNotFoundError

    def _load_cifar10(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        data = {
            'classes': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
            'x_train': x_train,
            'y_train': y_train.squeeze(),
            'x_test': x_test,
            'y_test': y_test.squeeze()
        }
        return data

    def _load_standford_online_product(self, datatype):
        train_datagen = ImageDataGenerator(dtype=np.uint8)
        test_datagen = ImageDataGenerator(dtype=np.uint8)

        data_path = os.path.join(self._data_root, datatype, 'standford_online_products')
        train_data_path = os.path.join(data_path, 'train')
        test_data_path = os.path.join(data_path, 'test')

        train_generator = train_datagen.flow_from_directory(train_data_path, class_mode='sparse')
        test_generator = test_datagen.flow_from_directory(test_data_path, class_mode='sparse')

        x_train, y_train = self._convert_generator_to_data(train_generator)
        x_test, y_test = self._convert_generator_to_data(test_generator)

        classes = [[] for i in range(train_generator.num_classes)]
        for (name, index) in train_generator.class_indices.items():
            classes[index] = name

        dataset = dict()
        dataset['classes'] = classes
        dataset['x_train'] = x_train
        dataset['y_train'] = y_train
        dataset['x_test'] = x_test
        dataset['y_test'] = y_test

        return dataset

    def _load_vehicles(self, datatype):
        train_datagen = ImageDataGenerator(dtype=np.uint8)
        test_datagen = ImageDataGenerator(dtype=np.uint8)

        data_path = os.path.join(self._data_root, datatype, 'vehicles')
        train_data_path = os.path.join(data_path, 'train')
        test_data_path = os.path.join(data_path, 'test')

        train_generator = train_datagen.flow_from_directory(train_data_path, class_mode='sparse')
        test_generator = test_datagen.flow_from_directory(test_data_path, class_mode='sparse')

        x_train, y_train = self._convert_generator_to_data(train_generator)
        x_test, y_test = self._convert_generator_to_data(test_generator)

        classes = [[] for i in range(train_generator.num_classes)]
        for (name, index) in train_generator.class_indices.items():
            classes[index] = name

        dataset = dict()
        dataset['classes'] = classes
        dataset['x_train'] = x_train
        dataset['y_train'] = y_train
        dataset['x_test'] = x_test
        dataset['y_test'] = y_test

        return dataset

    def _save_pickle(self, data, dataset_name, dataset_type):
        pickle_file_name = dataset_name + '.pickle'
        pickle_file_path = os.path.join(self._data_root, dataset_type, pickle_file_name)
        try:
            f = open(pickle_file_path, 'wb')
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            f.close()
        except Exception as e:
            print('Unable to save data to', pickle_file_path, ':', e)
            raise

    def _convert_generator_to_data(self, generator):
        x = np.ndarray(shape=((generator.n,) + generator.image_shape), dtype=np.uint8)
        y = np.ndarray(shape=generator.n, dtype=np.uint8)
        for i in range(generator.__len__()):
            x_batch, y_batch = generator.next()
            start_index = i * generator.batch_size
            end_index = (i + 1) * generator.batch_size
            x[start_index:end_index] = x_batch
            y[start_index:end_index] = y_batch

        return x, y