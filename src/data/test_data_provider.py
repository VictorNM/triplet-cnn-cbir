import unittest
from . import data_provider


class TestDataProvider(unittest.TestCase):
    def test_load_default(self):
        dataset = data_provider.load()
        print(dataset['classes'])
        print(dataset['x_train'].shape)
        print(dataset['x_valid'].shape)
        print(dataset['x_test'].shape)

    def test_load_standford_online_products(self):
        dataset = data_provider.load(dataset_name=data_provider.STANDFORD_ONLINE_PRODUCTS)
        print(dataset['classes'])
        print(dataset['triplet_index_train'].shape)
        print(dataset['triplet_index_valid'].shape)
        print(dataset['triplet_index_test'].shape)

        del dataset
        import gc
        gc.collect()

