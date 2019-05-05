import numpy as np
from sklearn.cluster import KMeans


def evaluate_classifier(classifier, x, y, config):
    # simply call keras model evaluate method
    score = classifier.evaluate(
        x=x,
        y=y,
        batch_size=config.batch_size
    )
    return score


def evaluate_extractor(extractor, dataset, config):
    """
    Evaluate using:
    - similarity_precision: percentage of triplets being correctly ranked.
        Given a triplet t_i = (a_i, p_i, n_i) where a_i: anchor, p_i: positive, n_i: negative
        t_i is correctly ranked if: distance(a_i, p_i) < distance(a_i, n_i)
    - mean_category_precision_at_top_K: mean of percentage of positive / top N results
    - mAP
    """
    x_train, y_train = dataset['x_train'], dataset['y_train']
    x_test, y_test = dataset['x_test'], dataset['y_test']
    classes = dataset['classes']

    return _mAP(extractor, x_train, y_train, x_test, y_test, num_classes=len(classes), k=5)


def _mAP(extractor, x_train, y_train, x_test, y_test, num_classes, k=10):
    x_train_vectors = extractor.predict(x_train)
    x_test_vectors = extractor.predict(x_test)

    kmeans = _create_kmeans(x_train_vectors, n_clusters=num_classes)
    test_kmeans_labels = kmeans.predict(x_test_vectors)

    APs = []
    num_tests = len(x_test_vectors)
    for i in range(num_tests):
        query_vector = x_test_vectors[i]
        query_label = y_test[i]
        query_kmeans_label = test_kmeans_labels[i]
        result_vectors, result_labels = _do_query(query_vector, query_kmeans_label, x_train_vectors, y_train, kmeans.labels_)
        distances = [_euclidean_distance(result_vector, query_vector) for result_vector in result_vectors]
        sorted_idx = np.argsort(distances)
        sorted_result_labels = result_labels[sorted_idx]
        AP = np.average([
            _category_precision_at_top_k(query_label, sorted_result_labels, i)
            for i in range(1, 5)
        ])
        APs.append(AP)

    return np.mean(APs)


def _do_query(query_vector, query_label, vectors_database, labels_databse, kmeans_labels):
    indexes = _where_equal(kmeans_labels, query_label)
    result_vectors = vectors_database[indexes]
    result_labels = labels_databse[indexes]

    return result_vectors, result_labels


def _create_kmeans(x, n_clusters):
    kmeans = KMeans(n_clusters)
    kmeans.fit(x)
    return kmeans


def _where_equal(arr, val):
    # return index of values in arr that equal to val
    return np.asanyarray(arr == val).nonzero()


def _get_sorted_indexes(query_vector, db_vectors):
    distances = []
    for db_vector in db_vectors:
        distances.append(_euclidean_distance(query_vector, db_vector))

    return np.argsort(distances)


def _euclidean_distance(a, b):
    return np.sqrt(np.sum(np.square(a - b)))


def _category_precision_at_top_k(query_label, sorted_labels, k):
    first = sorted_labels[:k]
    count_true = len(first[first == query_label])
    return count_true / k
