import sys
import os
import matplotlib.pyplot as plt
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.model_buider import ModelBuilder
from src.data.data_provider import DataProvider
from src.data.data_processor import DataProcessor

if __name__ == '__main__':

    dataset = DataProvider.load(DataProvider.CIFAR10)
    new_dataset = DataProcessor.scale()
    x_train = dataset['x_train']    
    plt.imshow(x_train[0,:,:,:])
    plt.show()
    