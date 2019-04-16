import keras
from keras.utils import to_categorical
class DataProcessor:
    @staticmethod
    def normalize(dataset, config):
        #TODO: implement this function
        return dataset

    @staticmethod
    def augment(dataset, config):
        # TODO: implement this function
        return dataset

    @staticmethod
    def scale(dataset):
       ##
       x_train = dataset['x_train'].astype('float32')
       x_test  = dataset['x_test'].astype('float32')
       y_train = dataset['y_train']
       y_test = dataset['y_test']

       x_train /= 255
       x_test /= 255
       num_classes = len(dataset['classes'])
       
       ## convert class
       y_train = to_categorical(dataset['y_train'],num_classes)
       y_test = to_categorical(dataset['y_test'],num_classes)
       return {
           'classes': dataset['classes'],
           'x_train' : x_train,
           'y_train' : y_train,
           'x_test' : x_test,
           'y_test' : y_test
       }
        
