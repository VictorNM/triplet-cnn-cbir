import gc
import os

import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.cluster import KMeans

from .utils import euclidean_distance, load_pickle, save_pickle, where_equal


class Database:
    def __init__(self, extractor, directory):
        self.extractor = extractor
        self.directory = directory
        self.kmeans = None
        self.image_shape = self.extractor.layers[0].input_shape[1:]

        self.processor = ImageDataGenerator(rescale=1./255, samplewise_center=True)        
        self.images_generator =  self.processor.flow_from_directory(
                                    directory, 
                                    target_size=self.image_shape[:-1], 
                                    classes=['images'], 
                                    shuffle=False,
                                    batch_size=320
                                )

        self.n_images = self.images_generator.n
        self.n_features = self.extractor.layers[-1].output_shape[1]

    def create_features_database(self, force=False):
        features_directory = os.path.join(self.directory, 'features')

        if os.path.exists(features_directory) and len(os.listdir(features_directory)) > 0 and not force:
            print('Features database already created at:', self.directory)
            return

        if not os.path.exists(features_directory):
            os.makedirs(features_directory)

        # create features
        features = np.empty(shape=(self.n_images, self.n_features))
        start_index = 0
        for i in range(len(self.images_generator)):
            gc.collect()
            x_batch, _ = self.images_generator.__getitem__(i)
            features_batch = self.extractor.predict(x_batch)            
            end_index = start_index + len(features_batch)
            features[start_index:end_index] = features_batch

            start_index += len(features_batch)
            print('Created: {} / {} features'.format(start_index, self.n_images))

        print('saving...')
        save_pickle(features, os.path.join(features_directory, 'features'))
        print('Created features database successfully')

    def create_kmeans(self, num_clusters):
        self.kmeans = KMeans(num_clusters)

        features_file_path = os.path.join(self.directory, 'features', 'features')
        features = load_pickle(features_file_path)
        
        self.kmeans.fit(features)
        print('Kmeans created')

    def query(self, query_image, use_kmeans=True, num_results=5):
        if num_results == 0:
            num_results = self.n_images

        # do preprocess
        resized_image = np.expand_dims(cv2.resize(query_image, self.image_shape[:-1]), axis=0).astype('float64')
        standardized_image = self.processor.standardize(resized_image)

        # extract feature
        query_feature = self.extractor.predict(standardized_image)

        # query from feature database
        features_file_path = os.path.join(self.directory, 'features', 'features')
        db_features = load_pickle(features_file_path)

        if use_kmeans:
            assert self.kmeans is not None, 'You have to call method create_kmeans first'
            return self._query_kmeans(query_feature, db_features, num_results)

        return self._query_normal(query_feature, db_features, num_results)

    def _query_kmeans(self, query_feature, db_features, num_results):
        query_cluster = self.kmeans.predict(query_feature)
        same_cluster_indices = where_equal(self.kmeans.labels_, query_cluster)

        # if num_results is less than num features in the cluster
        if len(same_cluster_indices >= num_results):
            same_cluster_features = db_features[same_cluster_indices]
            same_cluster_distances = euclidean_distance(query_feature, same_cluster_features)
            sorted_same_cluster_indices = np.argsort(same_cluster_distances)[:num_results]
            sorted_filenames = np.array(self.images_generator.filenames)[same_cluster_indices[sorted_same_cluster_indices]]
            return [cv2.imread(os.path.join(self.directory, filename), cv2.IMREAD_COLOR) 
                    for filename in sorted_filenames]

    def _query_normal(self, query_feature, db_features, num_results):
        distances = euclidean_distance(query_feature, db_features)
        sorted_indices = np.argsort(distances)[:num_results]
        sorted_filenames = np.array(self.images_generator.filenames)[sorted_indices]

        return [cv2.imread(os.path.join(self.directory, filename), cv2.IMREAD_COLOR) 
                for filename in sorted_filenames]
