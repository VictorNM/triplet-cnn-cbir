import numpy as np


def evaluate_classifier(classifier, x, y, config):
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
    train_feature_vectors = extractor.predict(x_train)
    test_feature_vectors = extractor.predict(x_test)

    num_results = 10    # TODO: read from config.num_results

    precisions = []
    for i, test_feature_vector in enumerate(test_feature_vectors):
        distances = []
        for train_feature_vector in train_feature_vectors:
            distance = _euclidean_distance(test_feature_vector, train_feature_vector)
            distances.append(distance)

        sorted_indexes = np.argsort(distances)
        sorted_labels = y_train[sorted_indexes]
        precision = _category_precision_at_top_k(y_test[i], sorted_labels, num_results)
        precisions.append(precision)

    mean_category_precision = np.average(precisions)
    return 0, mean_category_precision


def _euclidean_distance(a, b):
    return np.sqrt(np.sum(np.square(a - b)))


def _category_precision_at_top_k(query_label, sorted_labels, k=5):
    first = sorted_labels[:k]
    count_true = len(first[first == query_label])
    return count_true / k
