class ModelConfig(object):
    def __init__(self):
        self.name = None
        self.optimizer = None
        self.batch_size = None
        self.epochs = None
        self.learning_rate = None
        self.validation_split = None

    def __str__(self):
        return str(self.__dict__)
