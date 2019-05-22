import unittest

from keras.models import load_model

from src.database_tmp import Database
from src import visualization
import cv2
import matplotlib.pyplot as plt


class TestDatabase(unittest.TestCase):
    def test_init_database(self):
        extractor = load_model(
            '/home/victor/Learning/bku/dissertation/implementation/models/worked-triplet-loss-0.81_0.87.h5'
        )
        directory = '/home/victor/Learning/bku/dissertation/implementation/database/stanford_online_products'
        db = Database(extractor, directory)

        assert db.image_shape == (224, 224, 3)

    def test_create_features_database(self):
        extractor = load_model(
            '/home/victor/Learning/bku/dissertation/implementation/models/worked-triplet-loss-0.81_0.87.h5'
        )
        directory = '/home/victor/Learning/bku/dissertation/implementation/database/stanford_online_products'
        db = Database(extractor, directory)

        db.create_features_database(force=True)

    def test_create_kmeans(self):
        extractor = load_model(
            '/home/victor/Learning/bku/dissertation/implementation/models/worked-triplet-loss-0.81_0.87.h5'
        )
        directory = '/home/victor/Learning/bku/dissertation/implementation/database/stanford_online_products'
        db = Database(extractor, directory)
        db.create_kmeans(2)

    def test_query_kmeans(self):
        extractor = load_model(
            '/home/victor/Learning/bku/dissertation/implementation/models/worked-triplet-loss-0.81_0.87.h5'
        )
        directory = '/home/victor/Learning/bku/dissertation/implementation/database/stanford_online_products'
        db = Database(extractor, directory, samplewise_center=False)
        db.create_kmeans(2)
        query_image = cv2.imread(
            '/home/victor/Learning/bku/dissertation/implementation/database/stanford_online_products/images/111169942421_0.JPG',
            cv2.IMREAD_COLOR
        )
        images = db.query(query_image, use_kmeans=True, num_results=5)

        visualization.show_single_image(query_image)
        plt.show()
        for i in range(5):
            visualization.show_single_image(images[i])
            plt.show()

    def test_query_normal(self):
        extractor = load_model(
            '/home/victor/Learning/bku/dissertation/implementation/models/worked-triplet-loss-0.81_0.87.h5'
        )
        directory = '/home/victor/Learning/bku/dissertation/implementation/database/stanford_online_products'
        db = Database(extractor, directory, samplewise_center=False)
        query_image = cv2.imread(
            '/home/victor/Learning/bku/dissertation/implementation/database/stanford_online_products/images/111598451635_2.JPG',
            cv2.IMREAD_COLOR
        )
        images = db.query(query_image, use_kmeans=False, num_results=5)

        assert len(images) == 5

        visualization.show_single_image(query_image)
        plt.show()
        for i in range(5):
            visualization.show_single_image(images[i])
            plt.show()
