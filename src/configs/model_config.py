class ModelConfig(object):
    def __init__(self, config_dict=None):
        if config_dict is None:
            config_dict = {}

        self._parse(config_dict)

    def __str__(self):
        return str(self.__dict__)

    def _parse(self, config_dict):
        default = ModelConfig._default_config()
        self.name = config_dict.get('name', default['name'])

    @staticmethod
    def _default_config():
        return {
            'name': 'custom'
        }