import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.data_processor import DataProcessor

if __name__ == '__main__':
    dataset = ...
    config = ...new DataConfig()
    DataProcessor.normalize()