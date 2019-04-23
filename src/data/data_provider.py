import os
from keras.datasets import mnist, cifar10
import numpy as np
from six.moves import cPickle as pickle


class DataProvider:
    def __init__(self, data_root):
        self._data_root = data_root

    def load(self, dataset_name, dataset_type, train_samples=0, test_samples=0):
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
            dataset = self._load_from_pickle_file(dataset_name, dataset_type)
        except FileNotFoundError:
            if dataset_name == 'mnist':
                dataset = self._load_mnist()

            if dataset_name == 'cifar-10':
                dataset = self._load_cifar10()

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

    def _load_mnist(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        data = {
            'classes': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
            'x_train': np.expand_dims(x_train, -1),
            'y_train': y_train,
            'x_test': np.expand_dims(x_test, -1),
            'y_test': y_test
        }
        return data

    def _load_cifar10(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        data = {
            'classes': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
            'x_train': x_train,
            'y_train': y_train,
            'x_test': x_test,
            'y_test': y_test
        }
        return data

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
