import numpy as np


def evaluate_classifier(classifier, x, y, config):
    # simply call keras model evaluate method
    score = classifier.evaluate(
        x=x,
        y=y,
        batch_size=config.batch_size
    )
    return score


def evaluate_extractor(extractor, x_train, y_train, x_test, y_test, config):
    """
    Evaluate using:
    - similarity_precision: percentage of triplets being correctly ranked.
        Given a triplet t_i = (a_i, p_i, n_i) where a_i: anchor, p_i: positive, n_i: negative
        t_i is correctly ranked if: distance(a_i, p_i) < distance(a_i, n_i)
    - mean_category_precision_at_top_K: mean of percentage of positive / top N results
    - mAP

    :param extractor:
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param config:
    :return:
    """
    mAP = _mAP(extractor, x_train, y_train, x_test, y_test, k=10)
    return mAP


def _mAP(extractor, x_train, y_train, x_test, y_test, k=10):
    train_vectors = extractor.predict(x_train)
    test_vectors = extractor.predict(x_test)

    return np.average([
        _ap(
            query_vector=test_vector,
            db_vectors=train_vectors,
            query_label=y_test[i],
            db_labels=y_train,
            k=k
        )
        for i, test_vector in enumerate(test_vectors)
    ])


def _ap(query_vector, db_vectors, query_label, db_labels, k):
    sorted_db_labels = db_labels[_get_sorted_indexes(query_vector, db_vectors)]
    return np.average([
        _category_precision_at_top_k(query_label, sorted_db_labels, i)
        for i in range(1, k+1)
    ])


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
