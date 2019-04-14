import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.model_buider import ModelBuilder
from src.data.data_provider import DataProvider

if __name__ == '__main__':
    dataset = DataProvider.load(DataProvider.MNIST)
    net = ModelBuilder.load('vgg16')
    net.fit(dataset['x_train'], dataset['y_train'], epochs=2)
