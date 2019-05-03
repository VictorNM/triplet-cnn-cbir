import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
from data_provider import DataProvider

from configs.data_config  import DataConfig


class TestDataProvider(unittest.TestCase):
    def test_save_cifar_10(self):
        dataset = DataProvider.load('cifar-10')

    def test_load_cifar_10(self):
        data_root = 'D:\PythonProject\thesis\dataset'
        data_config = DataConfig()
        data_config.dataset_name = 'cifar-10'
        data_config.dataset_type = 'raw'
        data_provider = DataProvider(data_root, data_config)
        cifar = data_provider.load()
        print(cifar['classes'])

    def test_load_mnist(self):
        data_root = 'D:\PythonProject\thesis\dataset'
        data_config = DataConfig()
        data_config.dataset_name = 'mnist'
        data_config.dataset_type = 'raw'
        data_provider = DataProvider(data_root, data_config)
        mnist = data_provider.load()
        print(mnist['classes'])

if __name__ == '__main__':
    unittest.main()