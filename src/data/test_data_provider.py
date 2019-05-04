import unittest
from .data_provider import DataProvider
import matplotlib.pyplot as plt


class TestDataProvider(unittest.TestCase):
    def test_load_vehicles(self):
        data_provider = DataProvider(data_root='/home/victor/Learning/bku/dissertation/implementation/data')
        dataset = data_provider.load('vehicles', 'raw')
        print(dataset['classes'])
        print(dataset['x_test'].shape)
        print(dataset['y_train'][0])
        plt.imshow(dataset['x_train'][0])
        plt.show()
