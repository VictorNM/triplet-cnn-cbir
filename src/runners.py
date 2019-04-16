import os
from .runner import Runner


class Runners:
    def __init__(self, env='local', configs_dir=None):
        self.runners = Runners._init_runners(env, configs_dir)

    @staticmethod
    def _init_runners(env='local', configs_dir=None):
        if configs_dir is None:
            if env == 'local':
                configs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../configs'))
            elif env == 'colab':
                configs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../configs'))

        if not os.path.isabs(configs_dir) or not os.path.isdir(configs_dir):
            raise ValueError("configs_dir must be an absolute path to a directory")

        runners = {}
        for config_file_name in os.listdir(configs_dir):
            filename, file_extension = os.path.splitext(config_file_name)
            if file_extension != '.json':
                print(config_file_name + " is an invalid config file, skipped")
                continue

            runners[filename] = Runner(env, config_file_name, configs_dir)

        return runners
