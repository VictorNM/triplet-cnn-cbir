import os
import numpy as np
from math import inf
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence


class TripletGenerator(Sequence):
    def __init__(self, extractor, margin, directory, batch_size=16):
        self.extractor = extractor
        self.extractor._make_predict_function()
        self.margin = margin

        classes = os.listdir(directory)
        datagen = ImageDataGenerator(rescale=1./255, samplewise_center=True)

        input_size = extractor.layer[0].input_shape[1:-1]

        self.gen0 = datagen.flow_from_directory(
            directory=directory,
            input_size=input_size,
            batch_size=batch_size,
            classes=[classes[0]]
        )

        self.gen1 = datagen.flow_from_directory(
            directory=directory,
            input_size=input_size,
            batch_size=batch_size,
            classes=[classes[1]]
        )

    def __len__(self):
        return min(self.gen0.__len__(), self.gen1.__len__())

    def __getitem__(self, idx):
        x0_batch, y0_batch = self.gen0.__getitem__(idx)
        x1_batch, y1_batch = self.gen1.__getitem__(idx)

        y0_batch = np.argmax(y0_batch, axis=1)
        y1_batch = np.argmax(y1_batch, axis=1) + 1

        x_batch = np.concatenate((x0_batch, x1_batch))
        y_batch = np.concatenate((y0_batch, y1_batch))

        feature_batch = self.extractor.predict(x_batch)

        triplet_index_batch = self.get_triplet_index_hard(feature_batch, y_batch)
        a, p, n = self.get_triplets_images(triplet_index_batch, x_batch)

        return [a, p, n], None

    def on_epoch_end(self):
        self.gen0.reset()
        self.gen1.reset()

    def get_triplet_index_hard(self, features, labels):
        triplets = []

        for i in range(len(labels)):
            max_pos_d = 0
            max_pos_idx = None
            min_neg_d = inf
            min_neg_idx = None

            for j in range(len(labels)):
                if j == i:  # ignore the same image
                    continue

                distance = np.sqrt(
                    np.sum(np.square(features[i] - features[j])))

                if labels[j] == labels[i]:  # positive
                    if distance > max_pos_d:
                        max_pos_d = distance
                        max_pos_idx = j

                else:  # negative
                    if distance < min_neg_d:
                        min_neg_d = distance
                        min_neg_idx = j

            if max_pos_d + self.margin >= min_neg_d:
                triplets.append([i, max_pos_idx, min_neg_idx])

        return triplets

    def get_triplets_images(self, triplets_index, x):
        num_triplets = len(triplets_index)
        image_shape = x.shape[1:]
        anchors = np.empty(shape=(0,) + image_shape)
        positives = np.empty(shape=(0,) + image_shape)
        negatives = np.empty(shape=(0,) + image_shape)

        for i in range(num_triplets):
            ai, pi, ni = triplets_index[i]

            if pi is None or ni is None:
                continue

            a = x[ai]
            p = x[pi]
            n = x[ni]

            anchors = np.append(anchors, np.expand_dims(a, axis=0), axis=0)
            positives = np.append(positives, np.expand_dims(p, axis=0), axis=0)
            negatives = np.append(negatives, np.expand_dims(n, axis=0), axis=0)

        return anchors, positives, negatives
