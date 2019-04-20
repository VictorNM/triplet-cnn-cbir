import keras
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import mode


class Model:
    def __init__(self):
        self.cnn_classifier = None
        self.final_extractor = None
        self.cnn_extractor = None
        self.cnn_feature_layer_name = None

        self._cnnIsTrained = False
        self._extractorsIsMake = False

    def fit_cnn_classifier(self, x_train, y_train, num_classes):
        y_train = keras.utils.to_categorical(y_train, num_classes)
        self.cnn_classifier.fit(x_train, y_train)
        self._cnnIsTrained = True

    def evaluate_cnn_classifier(self, x_test, y_test, num_classes):
        y_test = keras.utils.to_categorical(y_test, num_classes)
        print(self.cnn_classifier.evaluate(x_test, y_test))

    def make_extractors(self):
        assert self._cnnIsTrained, "Must train CNN first"
        self.cnn_extractor = keras.models.Model(
            inputs=self.cnn_classifier.input,
            outputs=self.cnn_classifier.get_layer(self.cnn_feature_layer_name).output)

        for layer in self.cnn_extractor.layers:
            layer.trainable = False
        tmp = self.cnn_extractor.output
        feature_layer = keras.layers.Dense(32)(tmp)
        self.final_extractor = keras.models.Model(
            inputs=self.cnn_extractor.input,
            outputs=feature_layer
        )
        # TODO: self.final_extractor.compile()
        self._extractorsIsMake = True

    def evaluate_cnn_extractor(self, x_train, y_train, x_test, y_test):
        train_embeddings = self.cnn_extractor.predict(x_train)
        test_embeddings = self.cnn_extractor.predict(x_test)
        print("done predict")

        kmeans = KMeans(n_clusters=10, random_state=0).fit(train_embeddings)
        print("done kmean cluster")

        db = {}
        for i in range(10):
            db[i] = {}
            mask = (kmeans.labels_ == i)
            db[i]['embeddings'] = train_embeddings[mask]
            db[i]['labels'] = y_train[mask]

        print("done create data")

        precisions = []
        for i, test_embedding in enumerate(test_embeddings):
            cluster_label = kmeans.predict(test_embedding.reshape(1, -1))
            results = db[cluster_label[0]]
            distances = []
            for embedding in results['embeddings']:
                d = Model._euclidean_distance(test_embedding, embedding)
                distances.append(d)

            sorted_indexes = np.argsort(distances)
            sorted_labels = results['labels'][sorted_indexes]
            precision = Model._precision(y_test[i], sorted_labels, n=5)
            precisions.append(precision)

        print("All precisions", precisions)
        print("Average precision:", np.average(precisions))

    @staticmethod
    def _euclidean_distance(a, b):
        return np.sqrt(np.sum(np.square(a - b)))

    @staticmethod
    def _precision(query_label, sorted_labels, n=5):
        first = sorted_labels[:n]
        count_true = len(first[first == query_label])
        return count_true / n
