import unittest
from . import data_provider


class TestDataProvider(unittest.TestCase):
    def test_load_default(self):
        dataset = data_provider.load()
        print(dataset['classes'])
        print(dataset['x_train'].shape)
        print(dataset['x_valid'].shape)
        print(dataset['x_test'].shape)

    def test_load_vehicles(self):
        dataset = data_provider.load(dataset_name='vehicles')
        print(dataset['classes'])
        print(dataset['x_train'].shape)
        print(dataset['x_valid'].shape)
        print(dataset['x_test'].shape)

