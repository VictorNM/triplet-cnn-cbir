import gc
import os

import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from .utils import euclidean_distance, load_pickle, save_pickle


class Database:
    def __init__(self, extractor, directory):
        self.extractor = extractor
        self.directory = directory
        self.image_shape = self.extractor.layers[0].input_shape[1:]

        self.processor = ImageDataGenerator(rescale=1./255, samplewise_center=True)
        
        self.images_generator =  self.processor.flow_from_directory(
                                    directory, 
                                    target_size=self.image_shape[:-1], 
                                    classes=['images'], 
                                    shuffle=False,
                                    batch_size=320
                                )

    def create_features_database(self, force=False):
        features_directory = os.path.join(self.directory, 'features')

        if os.path.exists(features_directory) and len(os.listdir(features_directory)) > 0 and not force:
            print('Features database already created at:', self.directory)
            return

        if not os.path.exists(features_directory):
            os.makedirs(features_directory)

        num_features_created = 0
        for i in range(len(self.images_generator)):
            gc.collect()
            x_batch, _ = self.images_generator.__getitem__(i)
            features_batch = self.extractor.predict(x_batch)

            save_pickle(features_batch, os.path.join(features_directory, 'features_{}'.format(i)))
            num_features_created += len(features_batch)
            print('Created: {} / {} features'.format(num_features_created, self.images_generator.n))

        print('Created features database successfully')

    def query(self, query_image, num_results=5):
        # do preprocess
        resized_image = np.expand_dims(cv2.resize(query_image, self.image_shape[:-1]), axis=0).astype('float64')

        standardized_image = self.processor.standardize(resized_image)

        # extract feature
        query_feature = self.extractor.predict(standardized_image)

        # query from feature database
        features_directory = os.path.join(self.directory, 'features')
        distances = np.empty(shape=self.images_generator.n)
        start_index = 0
        for i, features_file in enumerate(os.listdir(features_directory)):
            features_batch = load_pickle(os.path.join(features_directory, features_file))
            distances_batch = euclidean_distance(features_batch, query_feature)
            end_index = start_index + len(distances_batch)
            distances[start_index:end_index] = distances_batch
            start_index += len(distances_batch)

        sorted_indices = np.argsort(distances)
        if num_results > 0:
            sorted_indices = sorted_indices[:num_results]

        # map to images database
        sorted_filenames = np.array(self.images_generator.filenames)[sorted_indices]
        results = [cv2.imread(os.path.join(self.directory, filename), cv2.IMREAD_COLOR) 
                    for filename in sorted_filenames]

        return results
