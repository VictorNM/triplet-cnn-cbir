import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.model_buider import ModelBuilder
from src.data.data_provider import DataProvider
from src.data.data_processor import DataProcessor

if __name__ == '__main__':
    # dataset = DataProvider.load(DataProvider.MNIST)
    dataset = DataProvider.load(DataProvider.CIFAR10)
    dataset = DataProcessor.scale(dataset)
    x_train = dataset['x_train']
    print(x_train[0])