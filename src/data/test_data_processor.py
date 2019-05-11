import unittest
import numpy as np
from . import data_processor


class TestDataProcessor(unittest.TestCase):
    def test_data_processor(self):
        x = np.random.randint(0, 255, size=(1, 100, 100, 3))
        print(x)
        x = data_processor.normalize(x, input_shape=(50, 50, 3))
        print(x)
        augmentation = {
            "rotation_range": 20,
            "horizontal_flip": True
        }
        x, y = data_processor.augment(x, augmentation)
        print(x)
