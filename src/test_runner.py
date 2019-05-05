import unittest
from .runner import Runner
from .configs.config import Config


class TestRunner(unittest.TestCase):
    def test_default(self):
        config = Config.from_file()
        runner = Runner(config=config)

        # Prepare data
        runner.load_data(train_samples=5, test_samples=1)
        runner.preprocess_data()

        # CNN classifier
        runner.build_cnn_classifier()
        runner.summary_cnn_classifier()
        runner.train_cnn_classifier()
        runner.evaluate_cnn_classifier()

        # CNN extractor
        runner.build_cnn_extractor()
        runner.evaluate_cnn_extractor()

    def test_config_from_dict(self):
        config_dict = {
            "data_config": {
                "dataset_name": "vehicles",
                "data_augmentation": {
                    "rotation_range": 20,
                    "horizontal_flip": True
                },
                "input_shape": [128, 128, 3]
            },
            "model_config": {
                "name": "custom"
            },
            "training_config": {
                "batch_size": 32,
                "epochs": 5,
                "learning_rate": 0.01,
                "validation_split": 0.3
            }
        }
        config = Config.from_dict(config_dict)
        runner = Runner(config=config)

        # Prepare data
        runner.load_data(train_samples=5, test_samples=1)
        runner.preprocess_data()

        # CNN classifier
        runner.build_cnn_classifier()
        runner.summary_cnn_classifier()
        runner.train_cnn_classifier()
        runner.evaluate_cnn_classifier()

        # CNN extractor
        runner.build_cnn_extractor()
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
