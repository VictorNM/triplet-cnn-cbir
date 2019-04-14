import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import unittest
from src.configs.configs import Configs


class TestConfigs(unittest.TestCase):
    def test_configs(self):
        configs = Configs()
        print(configs.configs['config'].model_config.batch_size)
