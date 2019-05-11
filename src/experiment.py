import numpy as np
from sklearn.cluster import KMeans
import time


def evaluate_classifier(classifier, x, y, config):
    # simply call keras model evaluate method
    score = classifier.evaluate(
        x=x,
        y=y,
        batch_size=config.batch_size
    )
    return score


def evaluate_extractor(extractor, dataset, mode, top_k=30):
    """
    Evaluate using:
    - similarity_precision: percentage of triplets being correctly ranked.
        Given a triplet t_i = (a_i, p_i, n_i) where a_i: anchor, p_i: positive, n_i: negative
        t_i is correctly ranked if: distance(a_i, p_i) < distance(a_i, n_i)
    - mean_category_precision_at_top_K: mean of percentage of positive / top N results
    - mAP
    """
    classes = dataset['classes']
    x_train, y_train = dataset['x_train'], dataset['y_train']
    if mode == 'valid':
        x_test, y_test = dataset['x_valid'], dataset['y_valid']
    elif mode == 'test':
        x_test, y_test = dataset['x_test'], dataset['y_test']
    else:
        raise ValueError("Invalid param 'mode'")

    start = time.time()
    similarity_precision = _similarity_precision(extractor, x_test, y_test)
    print('Time evaluate similarity', time.time() - start)
    start = time.time()
    mAP = _mAP(extractor, x_train, y_train, x_test, y_test, num_classes=len(classes), k=top_k)
    print('Time evaluate mAp', time.time() - start)

    return similarity_precision, mAP


def _similarity_precision(extractor, x, y):
    features = extractor.predict(x)
    num_valid_triplets = _count_valid_triplets(features, y)

    return num_valid_triplets / features.shape[0]


def _count_valid_triplets(features, labels):
    num_valid_triplets = 0
    for a in range(features.shape[0]):
        for p in range(features.shape[0]):
            if p == a or labels[p] != labels[a]:
                continue
            for n in range(features.shape[0]):
                if (a == n) or (p == n):
                    continue
                if labels[a] == labels[n]:
                    continue
                if _is_valid_triplet(features[a], features[p], features[n]):
                    num_valid_triplets += 1

    return num_valid_triplets


def _is_valid_triplet(a, p, n):
    return _euclidean_distance(a, p) < _euclidean_distance(a, n)


def _mAP(extractor, x_train, y_train, x_test, y_test, num_classes, k):
    x_train_vectors = extractor.predict(x_train)
    x_test_vectors = extractor.predict(x_test)

    kmeans = _create_kmeans(x_train_vectors, n_clusters=num_classes)
    test_kmeans_labels = kmeans.predict(x_test_vectors)
    num_tests = len(x_test_vectors)

    return np.mean([
        _AP(
            query_vector=x_test_vectors[i],
            query_label=y_test[i],
            query_kmeans_label=test_kmeans_labels[i],
            kmeans_labels=kmeans.labels_,
            x_train_vectors=x_train_vectors,
            y_train=y_train,
            k=k
        )
        for i in range(num_tests)
    ])


def _AP(query_vector, query_label, query_kmeans_label, kmeans_labels, x_train_vectors, y_train, k):
    result_vectors, result_labels = _do_query(query_kmeans_label, kmeans_labels, x_train_vectors, y_train)
    distances = [_euclidean_distance(result_vector, query_vector) for result_vector in result_vectors]
    sorted_result_labels = result_labels[np.argsort(distances)]
    return np.average([
        _category_precision_at_top_k(query_label, sorted_result_labels, i)
        for i in range(1, k+1)
    ])


def _do_query(query_kmeans_label, kmeans_labels, vectors_database, labels_databsse):
    indexes = _where_equal(kmeans_labels, query_kmeans_label)
    result_vectors = vectors_database[indexes]
    result_labels = labels_databsse[indexes]

    return result_vectors, result_labels


def _create_kmeans(x, n_clusters):
    kmeans = KMeans(n_clusters)
    kmeans.fit(x)
    return kmeans


def _where_equal(arr, val):
    # return index of values in arr that equal to val
    return np.asanyarray(arr == val).nonzero()


def _euclidean_distance(a, b):
    return np.sqrt(np.sum(np.square(a - b)))


def _category_precision_at_top_k(query_label, sorted_labels, k):
    first = sorted_labels[:k]
    count_true = len(first[first == query_label])
    return count_true / k
