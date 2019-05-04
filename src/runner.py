import os
from keras.utils import to_categorical
from .configs import config
from .data.data_provider import DataProvider
from .model.models import Models
from . import training, experiment


class Runner:
    def __init__(self, data_root='../data', configs_dir='../configs', config_file_name='config.json'):
        # Environment & Configurations
        if os.path.isabs(data_root):
            self._data_root = data_root
        else:
            self._data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), data_root))
        self._config = Runner._load_config(config_file_name, configs_dir)
        self._data_provider = None

        # Data
        self._data_raw = None
        self._data_processed = None

        # Models
        self._model_cnn_classifier = None
        self._model_cnn_extractor = None
        self._model_deep_ranking_extractor = None

        # Database
        self._database = None

    def load_data(self, train_samples=0, test_samples=0):
        print("\nLOADING DATA...")
        data_root = self._data_root
        dataset_name = self._config.data_config.dataset_name
        dataset_type = 'raw'
        self._data_provider = DataProvider(data_root)
        self._data_raw = self._data_provider.load(dataset_name, dataset_type, train_samples, test_samples)

    def preprocess_data(self):
        print("\nPROCESSING DATA...")
        # TODO: use DataProcessor to implement this
        assert self._data_raw is not None
        self._data_processed = self._data_raw

    def construct_cnn_classifier(self):
        model_config = self._config.model_config
        input_shape = self._config.data_config.input_shape
        num_classes = len(self._data_processed['classes'])
        self._model_cnn_classifier = Models.make_cnn_classifier(model_config, input_shape, num_classes)

    def train_cnn_classifier(self):
        print("\nTRAINING CNN...")
        assert self._model_cnn_classifier is not None
        assert self._data_processed is not None

        x_train, y_train = self._data_processed['x_train'], self._data_processed['y_train']

        num_classes = len(self._data_processed['classes'])
        y_train = to_categorical(y_train, num_classes)

        self._model_cnn_classifier, train_history = training.train_classifier(
            classifier=self._model_cnn_classifier,
            x=x_train,
            y=y_train,
            config=self._config.model_config
        )

        return train_history

    def evaluate_cnn_classifier(self):
        print("\nEVALUATING CNN CLASSIFIER...")
        x_test, y_test = self._data_processed['x_test'], self._data_processed['y_test']

        num_classes = len(self._data_processed['classes'])
        y_test = to_categorical(y_test, num_classes)
        score = experiment.evaluate_classifier(
            classifier=self._model_cnn_classifier,
            x=x_test,
            y=y_test,
            config=self._config.model_config
        )

        print("score:", score)

    def construct_cnn_extractor(self):
        assert self._model_cnn_classifier is not None
        self._model_cnn_extractor = Models.make_cnn_extractor(self._model_cnn_classifier)

    def evaluate_cnn_extractor(self):
        print("\nEVALUATING CNN EXTRACTOR...")
        assert self._model_cnn_extractor is not None, "call construct_cnn_extractor() first"

        x_train, y_train = self._data_processed['x_train'], self._data_processed['y_train']
        x_test, y_test = self._data_processed['x_test'], self._data_processed['y_test']

        score = experiment.evaluate_extractor(
            extractor=self._model_cnn_extractor,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            config=self._config.model_config
        )

        print("score:", score)

    def construct_deep_ranking_extractor(self):
        assert self._model_cnn_extractor is not None, "call construct_cnn_extractor() first"

        self._model_deep_ranking_extractor = Models.make_deep_ranking_extractor(self._model_cnn_extractor)

    def train_deep_ranking_extractor(self):
        pass

    def evaluate_deep_ranking_extractor(self):
        pass

    def create_database(self):
        pass

    def query(self, input_images):
        pass

    def show_images(self, images):
        pass

    @staticmethod
    def _load_config(config_file_name, configs_dir):
        if not os.path.isabs(configs_dir):
            configs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), configs_dir))

        if not os.path.isabs(configs_dir) or not os.path.isdir(configs_dir):
            raise ValueError("configs_dir must be an absolute path to a directory")

        config_file_path = os.path.join(configs_dir, config_file_name)
        print("Reading config from:", config_file_path)
        return config.Config(config_file_path)
