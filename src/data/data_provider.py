import os
from keras.datasets import mnist, cifar10
import numpy as np
from six.moves import cPickle as pickle


class DataProvider:

    MNIST = 'mnist'
    CIFAR10 = 'cifar-10'

    def __init__(self, data_root, data_config):
        self.data_root = data_root
        self.data_config = data_config

    def load(self):
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

        dataset_name = self.data_config.dataset_name
        dataset = None
        if dataset_name == DataProvider.MNIST:
            dataset = self._load_mnist()

        if dataset_name == DataProvider.CIFAR10:
            dataset = self._load_cifar10()

        return dataset

    def _load_mnist(self):
        pickle_file_path = os.path.join(self.data_root, self.data_config.dataset_type, 'minst.pickle')
        if os.path.exists(pickle_file_path):
            data = DataProvider._load_pickle(pickle_file_path)
            return data

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        data = {
            'classes': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
            'x_train': np.expand_dims(x_train, -1),
            'y_train': y_train,
            'x_test': np.expand_dims(x_test, -1),
            'y_test': y_test
        }
        DataProvider._save_pickle(data, pickle_file_path)
        return data

    def _load_cifar10(self):
        pickle_file_path = os.path.join(self.data_root, self.data_config.dataset_type, 'cifar-10.pickle')
        if os.path.exists(pickle_file_path):
            data = DataProvider._load_pickle(pickle_file_path)
            return data

        # pickle file not existed, load from keras then save to pickle_file_path
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        data = {
            'classes': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
            'x_train': x_train,
            'y_train': np.reshape(y_train,y_train.shape[0]),
            'x_test': x_test,
            'y_test': np.reshape(y_test,y_test.shape[0])
        }
        DataProvider._save_pickle(data, pickle_file_path)
        return data

    @staticmethod
    def _save_pickle(data, pickle_file_path):
        try:
            f = open(pickle_file_path, 'wb')
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            f.close()
        except Exception as e:
            print('Unable to save data to', pickle_file_path, ':', e)
            raise

    @staticmethod
    def _load_pickle(pickle_file_path):
        with open(pickle_file_path, 'rb') as f:
            data = pickle.load(f)
            return data
