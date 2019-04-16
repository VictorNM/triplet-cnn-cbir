import os
from keras.utils import to_categorical
from .configs import config
from .data.data_provider import DataProvider
from .model.model_buider import ModelBuilder


class Runner:
    def __init__(self, env=None, config_file_name='config.json'):
        self.env = {
            "data_root": "../data",
            "configs_dir": "../configs"
        }
        if env is not None:
            self.env = env

        self.config = Runner._load_config(config_file_name, self.env['configs_dir'])
        self.data_raw = None
        self.data_processed = None
        self.model_cnn = None

        self.isPrepared = False
        self.isTrainedCNN = False

    def prepare(self):
        # load data
        self._load_data()

        # pre-process data
        self.data_processed = self.data_raw

        # build CNN
        self._load_cnn()

        self.isPrepared = True

    def prepare_cnn(self):
        pass

    def prepare_triplet(self):
        pass

    def analyze(self):
        pass

    def train_cnn(self):
        assert self.isPrepared

        x_train, y_train = self.data_processed['x_train'], self.data_processed['y_train']
        classes = self.data_processed['classes']

        self.model_cnn.fit(x_train, to_categorical(y_train, len(classes)))
        self.isTrainedCNN = True

    def evaluate(self):
        pass

    def save(self):
        pass

    def _load_data(self):
        if os.path.isabs(self.env['data_root']):
            data_root = self.env['data_root']
        else:
            data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), self.env['data_root']))

        data_provider = DataProvider(data_root, self.config.data_config)
        self.data_raw = data_provider.load()

    def _load_cnn(self):
        input_shape = self.data_processed['x_train'].shape[1:]
        num_classes = len(self.data_processed['classes'])
        self.model_cnn = ModelBuilder.load(self.config.model_config, input_shape, num_classes)

    @staticmethod
    def _load_config(config_file_name, configs_dir):
        if not os.path.isabs(configs_dir):
            configs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), configs_dir))

        if not os.path.isabs(configs_dir) or not os.path.isdir(configs_dir):
            raise ValueError("configs_dir must be an absolute path to a directory")

        config_file_path = os.path.join(configs_dir, config_file_name)
        print("Reading config from:", config_file_path)
        return config.Config(config_file_path)
