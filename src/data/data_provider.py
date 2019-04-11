from keras.datasets import mnist, cifar10
import numpy as np


class DataProvider:

    MNIST = 'mnist'
    CIFAR10 = 'cifar-10'

    @staticmethod
    def load(dataset_name):
        """

        :param dataset_name: name of the dataset
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
        if dataset_name == DataProvider.MNIST:
            dataset = DataProvider._load_mnist()

        if dataset_name == DataProvider.CIFAR10:
            dataset = DataProvider._load_cifar10()

        DataProvider._validate_return_value(dataset)

        return dataset

    @staticmethod
    def _load_mnist():
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        return {
            'classes': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
            'x_train': np.expand_dims(x_train, -1),
            'y_train': y_train,
            'x_test': np.expand_dims(x_test, -1),
            'y_test': y_test
        }

    @staticmethod
    def _load_cifar10():
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        return {
            'classes': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
            'x_train': x_train,
            'y_train': y_train,
            'x_test': x_test,
            'y_test': y_test
        }

    @staticmethod
    def _validate_return_value(dataset):
        """ Because some mistake can be make, so I add this function to prevent this """
        error_message = "DataProvider implementation error: "
        assert dataset is not None, error_message + "dataset is None"
        assert dataset['classes'] is not None, error_message + "dataset['classes'] is None"
        assert dataset['x_train'] is not None, error_message + "dataset['x_train'] is None"
        assert dataset['y_train'] is not None, error_message + "dataset['y_train'] is None"
        assert dataset['x_test'] is not None, error_message + "dataset['x_test'] is None"
        assert dataset['y_test'] is not None, error_message + "dataset['y_test'] is None"
        assert len(dataset['x_train'].shape) == 4, error_message + "x_train is not a 4-dimensions array"
        assert len(dataset['x_test'].shape) == 4, error_message + "x_test is not a 4-dimensions array"
        assert len(dataset['y_train'].shape) == 1, error_message + "y_train is not a 1-dimension array"
        assert len(dataset['y_test'].shape) == 1, error_message + "y_test is not a 1-dimension array"
        assert len(dataset['x_train']) == len(dataset['y_train']), error_message + "x_train, y_train don't have same length"
        assert len(dataset['x_test']) == len(dataset['x_test']), error_message + "x_test, y_test don't have same length"
