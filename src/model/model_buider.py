from keras.applications import vgg16


class ModelBuilder(object):
    @staticmethod
    def vgg16():
        return vgg16.VGG16()