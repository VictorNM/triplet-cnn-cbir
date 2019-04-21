import keras
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from keras.preprocessing.image import ImageDataGenerator
from configs.config import Config

class DataProcessor:

    @staticmethod
    def augment(config):
        # return datagen to add model 
        data_augmentation = config.data_config.data_augmentation
        datagen = ImageDataGenerator(**data_augmentation)
        return datagen

config = Config()
print(DataProcessor.augment(config))