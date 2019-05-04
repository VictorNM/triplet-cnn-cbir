class DataConfig:
    def __init__(self):
        self.dataset_name = None
        self.input_shape = None
        self.data_augmentation = None

    def __str__(self):
        return str(self.__dict__)
