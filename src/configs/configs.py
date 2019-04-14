import os
from .config import Config


class Configs:
    def __init__(self, config_dir='../../configs'):
        self.configs = {}
        abs_configs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), config_dir))
        print("Reading configuration files at: " + abs_configs_dir)
        config_file_names = os.listdir(abs_configs_dir)
        for config_file_name in config_file_names:
            filename, file_extension = os.path.splitext(config_file_name)
            if file_extension != '.json':
                print(config_file_name + " is an invalid config file, skipped")
                continue

            abs_config_file_path = os.path.abspath(os.path.join(abs_configs_dir, config_file_name))
            self.configs[filename] = Config(abs_config_file_path)
