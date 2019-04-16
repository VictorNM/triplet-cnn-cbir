import unittest
from .model_buider import ModelBuilder
from ..configs.config import Config


class TestModelBuilder(unittest.TestCase):
    def test_model_builder(self):
        model_config = Config().model_config
        model_config.name = 'custom'

        model = ModelBuilder.load(model_config, (32, 32, 3), 10)
        model.summary()
