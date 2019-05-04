import unittest
from .config import Config


class TestConfig(unittest.TestCase):
    def test_load_default_config(self):
        config = Config.from_file()
        print(config)

    def test_load_config_from_abs_path_file(self):
        config = Config.from_file('/home/victor/Learning/bku/dissertation/implementation/configs/config_2.json')
        print(config)
        assert config is not None

    def test_load_config_from_dict(self):
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
                "name": "custom",
                "optimizer": "Adam",
                "batch_size": 32,
                "epochs": 5,
                "learning_rate": 0.01,
                "validation_split": 0.3
            }
        }
        config = Config.from_dict(config_dict)
        print(config)
