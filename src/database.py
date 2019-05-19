import numpy as np
from sklearn.cluster import KMeans
import time
from . import utils


class KmeanDatabase:
    def __init__(self, extractor):
        self.extractor = extractor
        self.images = None
        self.features = None
        self.classes = None
        self.labels = None
        self.kmeans = None

    def create_database(self, images, labels, classes):
        print('Creating database with %d images...' % images.shape[0])
        start = time.time()
        self.images = images
        self.labels = labels
        self.classes = classes
        self.features = self.extractor.predict(images)

        num_classes = len(classes)
        self.kmeans = KMeans(num_classes)
        self.kmeans.fit(self.features)
        print('Database created in %ds' % (time.time() - start))

    def query(self, image, num_results=0):
        print('Querying...')
        start = time.time()
        query_feature = self.extractor.predict(np.expand_dims(image, axis=0))

        cluster_index = self.kmeans.predict(query_feature)
        result_index = utils.where_equal(self.kmeans.labels_, cluster_index)

        result_features = self.features[result_index]

        distances = utils.euclidean_distance(query_feature, result_features)
        sorted_result_index = result_index[np.argsort(distances)]
        print('Query finished in %ds' % (time.time() - start))

        result_images = self.images[sorted_result_index]
        result_labels = self.labels[sorted_result_index]

        if num_results > 0:
            return result_images[:num_results], result_labels[:num_results]

        return result_images, result_labels

    def get_real_labels(self, numeric_labels):
        return [self.classes[label] for label in numeric_labels]


class NormalDatabase:
    def __init__(self, extractor):
        self.extractor = extractor
        self.images = None
        self.features = None
        self.classes = None
        self.labels = None

    def create_database(self, images, labels, classes):
        print('Creating database with %d images...' % images.shape[0])
        start = time.time()
        self.images = images
        self.labels = labels
        self.classes = classes
        self.features = self.extractor.predict(images)

        print('Database created in %ds' % (time.time() - start))

    def query(self, image, num_results=0):
        print('Querying...')
        start = time.time()
        query_feature = self.extractor.predict(np.expand_dims(image, axis=0))

        distances = [utils.euclidean_distance(query_feature, db_feature) for db_feature in self.features]
        sorted_result_index = np.argsort(distances)
        print('Query finished in %ds' % (time.time() - start))

        result_images = self.images[sorted_result_index]
        result_labels = self.labels[sorted_result_index]

        if num_results > 0:
            return result_images[:num_results], result_labels[:num_results]

        return result_images, result_labels

    def get_real_labels(self, numeric_labels):
        return [self.classes[label] for label in numeric_labels]