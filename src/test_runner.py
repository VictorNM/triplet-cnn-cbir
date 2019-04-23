import unittest
from .runner import Runner


class TestRunner(unittest.TestCase):
    def test_runner(self):
        runner = Runner()
        runner.load_data(train_samples=100, test_samples=20)
        runner.preprocess_data()
        runner.construct_cnn_classifier()
        runner.train_cnn_classifier()
        runner.evaluate_cnn_classifier()
        runner.construct_cnn_extractor()
        runner.evaluate_cnn_extractor()

        # load data
        # preprocess data
        # analyze & visualize data
        # build classify CNN model
        # train with classify CNN model
        # evaluate classify CNN model
        # build CNN extractor from CNN classifier
        # evaluate CNN extractor (use K-Mean)
        # analyze & visualize data
        # build triplet model from CNN extractor
        # train triplet model
        # evaluate triplet model (use K-Mean)
        # analyze & visualize data
        # create DB (use K-Mean)
        # query (one image, many images)
        # GUI (web)
