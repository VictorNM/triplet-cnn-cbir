import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import unittest
from src.configs.configs import Configs


class TestConfigs(unittest.TestCase):
    def test_configs(self):
        configs = Configs()
        print(configs.configs['config'].model_config.batch_size)
        data_augmentation =  configs.configs['config'].data_config.data_augmentation
        print(data_augmentation)

test = TestConfigs()
test.test_configs()

