class TrainingConfig(object):
    def __init__(self, config_dict=None):
        if config_dict is None:
            config_dict = {}

        self._parse(config_dict)

    def __str__(self):
        return str(self.__dict__)

    def _parse(self, config_dict):
        default = TrainingConfig._default_config()
        self.optimizer = config_dict.get('optimizer', default['optimizer'])
        self.batch_size = config_dict.get('batch_size', default['batch_size'])
        self.epochs = config_dict.get('epochs', default['epochs'])
        self.validation_split = config_dict.get('validation_split', default['validation_split'])

    @staticmethod
    def _default_config():
        return {
            'optimizer': {
                'name': 'SGD'
            },
            'batch_size': 32,
            'epochs': 1,
            'validation_split': 0.3
        }
