import unittest
from .experiment import _count_valid_triplets, _is_valid_triplet
from .configs.config import Config
from .runner import Runner
import numpy as np


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

    def test_count_valid_triplets(self):
        features = np.array([
            [1, 1],
            [1, 2],
            [1, 10]
        ])
        labels = [1, 1, 2]
        assert _count_valid_triplets(features, labels) == 2

    def test_is_valid_triplet(self):
        a = np.array([1, 1])
        p = np.array([1, 2])
        n = np.array([1, 10])
        assert _is_valid_triplet(a, p, n)
