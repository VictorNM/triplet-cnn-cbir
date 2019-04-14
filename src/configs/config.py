import os
import json
from .model_config import ModelConfig
from .data_config import DataConfig


class Config:
    def __init__(self, config_file_path='../../configs/config.json'):
        self.data_config = None
        self.model_config = None

        if os.path.isabs(config_file_path):
            abs_config_file_path = config_file_path
        else:
            abs_config_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), config_file_path))

        with open(abs_config_file_path) as config_file:
            config_json = json.load(config_file)
            data_config, model_config = self._parse(config_json)
            self.data_config = data_config
            self.model_config = model_config

    def _parse(self, config_json):
        model_config = self._parse_model_config(config_json['model_config'])
        data_config = self._parse_data_config(config_json['data_config'])
        return data_config, model_config

    def _parse_model_config(self, model_config_json):
        model_config = ModelConfig()
        model_config.batch_size = model_config_json['batch_size']
        model_config.epochs = model_config_json['epochs']
        model_config.validation_split = model_config_json['validation_split']
        return model_config

    def _parse_data_config(self, data_config_json):
        data_config = DataConfig()
        return data_config