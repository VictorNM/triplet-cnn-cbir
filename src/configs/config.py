import os
import json
from .model_config import ModelConfig
from .data_config import DataConfig


class Config(object):
    def __init__(self):
        self.data_config = None
        self.model_config = None

    def __str__(self):
        return str({
            'data_config': str(self.data_config),
            'model_config': str(self.model_config)
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
        assert isinstance(config_dict, dict)
        assert 'data_config' in config_dict.keys()
        assert 'model_config' in config_dict.keys()

        return Config._parse(config_dict)


    @staticmethod
    def _parse(config_dict):
        config = Config()
        config.model_config = Config._parse_model_config(config_dict['model_config'])
        config.data_config = Config._parse_data_config(config_dict['data_config'])
        return config

    @staticmethod
    def _parse_model_config(model_config_dict):
        model_config = ModelConfig()
        if 'name' in model_config_dict:
            model_config.name = model_config_dict['name']
        if 'batch_size' in model_config_dict:
            model_config.batch_size = model_config_dict['batch_size']
        if 'epochs' in model_config_dict:
            model_config.epochs = model_config_dict['epochs']
        if 'learning_rate' in model_config_dict:
            model_config.learning_rate = model_config_dict['learning_rate']
        if 'validation_split':
            model_config.validation_split = model_config_dict['validation_split']
        return model_config

    @staticmethod
    def _parse_data_config(data_config_dict):
        data_config = DataConfig()
        data_config.dataset_name = data_config_dict['dataset_name']
        data_config.data_augmentation = data_config_dict['data_augmentation']
        data_config.input_shape = tuple(data_config_dict['input_shape'])
        return data_config
