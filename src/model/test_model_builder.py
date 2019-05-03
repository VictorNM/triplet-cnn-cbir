import unittest
from model_buider import ModelBuilder
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.model_config import ModelConfig


class TestModelBuilder(unittest.TestCase):
    def test_model_builder(self):
        model_config = ModelConfig()
        model_config.name = 'custom'

        model = ModelBuilder.load(model_config, (32, 32, 3), 10)
        model.summary()
if __name__ == "__main__":
    unittest.main()