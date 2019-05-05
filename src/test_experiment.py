import unittest
from .experiment import *
from .configs.config import Config
from .runner import Runner


class TestExperiment(unittest.TestCase):
    def test_kmeans(self):
        config = Config.from_dict()
        runner = Runner(config=config)
        runner.load_data(train_samples=10, test_samples=5)
        runner.preprocess_data()
        runner.build_cnn_classifier()
        runner.train_cnn_classifier()
        runner.build_cnn_extractor()
        runner.evaluate_cnn_extractor()
