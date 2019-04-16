from keras.applications import vgg16
from keras.models import Sequential

class ModelBuilder(object):
    @staticmethod
    def load(model_name, config):
        if model_name == 'vgg16':
            return ModelBuilder._vgg16()

    @staticmethod
    def _vgg16():
        return vgg16.VGG16()

