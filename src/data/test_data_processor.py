import unittest
from .data_processor import DataProcessor
from ..configs.data_config import DataConfig
import numpy as np


class TestDataProcessor(unittest.TestCase):
    def test_load_vehicles(self):
        config = DataConfig()
        config.data_augmentation = {
            "rescale": 2.0
        }
        data_processor = DataProcessor(config)
        x = np.random.randint(0, 255, size=(1, 3, 3, 3))
        print(x)
        x = data_processor.normalize(x)
        print(x)
        x, y = data_processor.augment(x, [0])
        print(x)
