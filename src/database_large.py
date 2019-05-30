import gc
import os
import shutil

import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from sklearn.cluster import KMeans

from .utils import euclidean_distance, load_pickle, save_pickle, where_equal


class Database:
    def __init__(self, extractor, directory):
        self.extractor = extractor
        self.extractor._make_predict_function()
        self.directory = directory
        self.kmeans = None
        self.features = None
        self.image_shape = self.extractor.layers[0].input_shape[1:]

        self.processor = ImageDataGenerator(rescale=1./255)
        self.images_generator = self.processor.flow_from_directory(
                                    directory, 
                                    target_size=self.image_shape[:-1], 
                                    classes=['images'], 
                                    shuffle=False,
                                    batch_size=160
                                )

        self.n_images = self.images_generator.n
        self.n_features = self.extractor.layers[-1].output_shape[1]

    def create_features_database(self, force=False):
        features_directory = os.path.join(self.directory, 'features')

        if os.path.exists(features_directory) and len(os.listdir(features_directory)) > 0 and not force:
            print('Features database already created at:', self.directory)
            return

        if os.path.exists(features_directory):
            shutil.rmtree(features_directory)

        os.makedirs(features_directory)

        # create features
        features = np.empty(shape=(self.n_images, self.n_features))
        start_index = 0
        for i in range(len(self.images_generator)):
            x_batch, _ = self.images_generator.__getitem__(i)
            features_batch = self.extractor.predict(x_batch)            
            end_index = start_index + len(features_batch)
            features[start_index:end_index] = features_batch

            start_index += len(features_batch)
            print('Created: {} / {} features'.format(start_index, self.n_images))

        self.features = features
        print('saving...')
        save_pickle(features, os.path.join(features_directory, 'features'))
        print('Created features database successfully')

    def load_features_database(self, features_filename):
        features_file_path = os.path.join(self.directory, 'features', features_filename)
        self.features = load_pickle(features_file_path)

    def create_kmeans(self, num_clusters):
        assert self.features is not None, "Load or create features first"
        self.kmeans = KMeans(num_clusters)

        self.kmeans.fit(self.features)
        print('Kmeans created')

    def load_image(self, image_path):
        return image.load_img(image_path, target_size=self.image_shape[:-1])

    def query(self, query_image, use_kmeans=True, num_results=5, return_path=False):
        assert self.features is not None, "Load or create features first"

        if num_results == 0:
            num_results = self.n_images

        # do preprocess
        standardized_image = self.processor.standardize(image.img_to_array(query_image))

        # extract feature
        query_feature = self.extractor.predict(np.expand_dims(standardized_image, axis=0))

        # query from feature database
        db_features = self.features

        if use_kmeans:
            assert self.kmeans is not None, 'You have to call method create_kmeans first'
            return self._query_kmeans(query_feature, db_features, num_results, return_path=return_path)

        return self._query_normal(query_feature, db_features, num_results, return_path=return_path)

    def _query_kmeans(self, query_feature, db_features, num_results, return_path=False):
        query_cluster = self.kmeans.predict(query_feature)
        same_cluster_indices = where_equal(self.kmeans.labels_, query_cluster)
        
        same_cluster_features = db_features[same_cluster_indices]
        same_cluster_distances = euclidean_distance(query_feature, same_cluster_features)
        sorted_same_cluster_indices = np.argsort(same_cluster_distances)[:num_results]
        sorted_filenames = np.array(self.images_generator.filenames)[same_cluster_indices[sorted_same_cluster_indices]]

        # because num of features in the same cluster are greater than num_results,
        # we don't need to query another clusters
        if len(same_cluster_indices) >= num_results:
            if return_path:
                return [os.path.join(self.directory, filename) for filename in sorted_filenames]

            return [self.load_image(os.path.join(self.directory, filename))
                    for filename in sorted_filenames]

        # query other clusters
        diff_cluster_indices = where_equal(self.kmeans.labels_, query_cluster)
        
        diff_cluster_features = db_features[diff_cluster_indices]
        diff_cluster_distances = euclidean_distance(query_feature, diff_cluster_features)
        sorted_diff_cluster_indices = np.argsort(diff_cluster_distances)[:num_results]
        sorted_diff_cluster_filenames = np.array(self.images_generator.filenames)[diff_cluster_indices[sorted_diff_cluster_indices]]

        sorted_filenames = np.concatenate((sorted_filenames, sorted_diff_cluster_filenames))

        if return_path:
            return [os.path.join(self.directory, filename) for filename in sorted_filenames]

        return [self.load_image(os.path.join(self.directory, filename))
                for filename in sorted_filenames]

    def _query_normal(self, query_feature, db_features, num_results, return_path=False):
        distances = euclidean_distance(query_feature, db_features)
        sorted_indices = np.argsort(distances)[:num_results]
        sorted_filenames = np.array(self.images_generator.filenames)[sorted_indices]

        if return_path:
            return [os.path.join(self.directory, filename) for filename in sorted_filenames]

        return [self.load_image(os.path.join(self.directory, filename))
                for filename in sorted_filenames]
