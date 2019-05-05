import os
import json
from .model_config import ModelConfig
from .data_config import DataConfig
from .training_config import TrainingConfig


class Config(object):
    def __init__(self):
        self.data_config = None
        self.model_config = None
        self.training_config = None

    def __str__(self):
        return str({
            'data_config': str(self.data_config),
            'model_config': str(self.model_config),
            'training_config': str(self.training_config)
        })

    @staticmethod
    def from_file(file_path='../../configs/config.json'):
        if os.path.isabs(file_path):
            abs_path = file_path
        else:
            abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), file_path))

        with open(abs_path) as config_file:
            config_dict = json.load(config_file)
            return Config._parse(config_dict)

    @staticmethod
    def from_dict(config_dict=None):
        return Config._parse(config_dict)

    @staticmethod
    def _parse(config_dict):
        config = Config()
        config.model_config = ModelConfig(config_dict.get('model_config'))
        config.data_config = DataConfig(config_dict.get('data_config'))
        config.training_config = TrainingConfig(config_dict.get('training_config'))

        return config
