import unittest
from .runner import Runner


class TestRunner(unittest.TestCase):
    def test_runner(self):
        runner = Runner()
        runner.prepare()
        runner.data_processed['x_train'] = runner.data_processed['x_train'][:5000]
        runner.data_processed['y_train'] = runner.data_processed['y_train'][:5000]
        runner.data_processed['x_test'] = runner.data_processed['x_test'][:1000]
        runner.data_processed['y_test'] = runner.data_processed['y_test'][:1000]
        runner.train_cnn()
        runner.evaluate_cnn()
        runner.make_extractors()
        runner.evaluate_cnn(mode='similarity')
