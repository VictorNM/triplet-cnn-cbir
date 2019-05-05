class DataConfig:
    def __init__(self, config_dict=None):
        if config_dict is None:
            config_dict = {}

        self._parse(config_dict)

    def __str__(self):
        return str(self.__dict__)

    def _parse(self, config_dict):
        default = DataConfig._default_config()
        self.dataset_name = config_dict.get('dataset_name', default['dataset_name'])
        self.data_augmentation = config_dict.get('data_augmentation', default['data_augmentation'])
        self.input_shape = tuple(config_dict.get('input_shape', default['input_shape']))

    @staticmethod
    def _default_config():
        return {
            'dataset_name': 'standford_online_products',
            'data_augmentation': {},
            'input_shape': [128, 128, 3]
        }
