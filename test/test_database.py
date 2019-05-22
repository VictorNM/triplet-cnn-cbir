import unittest
import os

from keras.models import load_model

from src.database_tmp import Database
from src.utils import load_pickle
import cv2


class TestDatabase(unittest.TestCase):
    def test_init_database(self):
        extractor = load_model('/home/victor/Learning/bku/dissertation/implementation/models/worked-triplet-loss-0.81_0.87.h5')
        directory = '/home/victor/Learning/bku/dissertation/implementation/database/stanford_online_products'
        db = Database(extractor, directory)

        assert db.image_shape == (224, 224, 3)

    def test_create_features_database(self):
        extractor = load_model('/home/victor/Learning/bku/dissertation/implementation/models/worked-triplet-loss-0.81_0.87.h5')
        directory = '/home/victor/Learning/bku/dissertation/implementation/database/stanford_online_products'
        db = Database(extractor, directory)

        db.create_features_database()

    def test_query(self):
        extractor = load_model('/home/victor/Learning/bku/dissertation/implementation/models/worked-triplet-loss-0.81_0.87.h5')
        directory = '/home/victor/Learning/bku/dissertation/implementation/database/stanford_online_products'
        db = Database(extractor, directory)
        query_image = cv2.imread('/home/victor/Learning/bku/dissertation/implementation/database/stanford_online_products/images/111169942421_0.JPG', cv2.IMREAD_COLOR)
        images = db.query(query_image, num_results=1)

        print(len(images))
        print(images[0].shape)