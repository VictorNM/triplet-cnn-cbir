import time

import numpy as np
from sklearn.cluster import KMeans

from . import utils


def mAP_normal(extractor, x, y):
    features = extractor.predict(x)

    return np.mean(
        [AP_normal(
            features=features,
            labels=y,
            index=i
        )
            for i in range(len(features))]
    )


def AP_normal(features, labels, index):
    query_feature = features[index]
    query_label = labels[index]

    db_features = np.delete(features, index, axis=0)
    db_labels = np.delete(labels, index, axis=0)

    total_relevents = len(utils.where_equal(db_features, query_label))

    # if db doesn't have relevant images, return 0
    if total_relevents == 0:
        return 0

    distances = utils.euclidean_distance(query_feature, db_features)
    sorted_index = np.argsort(distances)

    sorted_labels = db_labels[sorted_index]

    total_images = len(sorted_labels)
    tp = 0
    sum_precisions = 0

    for i in range(total_images):
        if sorted_labels[i] == query_label:
            tp += 1
            sum_precisions += tp / (i+1)

    return sum_precisions / total_relevents


def mAP_kmeans(extractor, x, y):
    features = extractor.predict(x)

    return np.mean([
        AP_kmeans(
            features=features,
            labels=y,
            index=i
        )
        for i in range(len(features))
    ])


def AP_kmeans(features, labels, index):
    query_feature = features[index]
    query_label = labels[index]

    db_features = np.delete(features, index, axis=0)
    db_labels = np.delete(labels, index, axis=0)

    total_relevents = len(utils.where_equal(db_labels, query_label))

    # if db doesn't have relevant images, return 0
    if total_relevents == 0:
        return 0

    kmeans = KMeans(n_clusters=len(set(db_labels)))
    kmeans.fit(db_features)
    kmeans_labels = kmeans.labels_
    kmeans_prediction = kmeans.predict(np.expand_dims(query_feature, axis=0))

    # all features in same cluster will be sorted and put to top
    # other features will also be sorted and put after
    same_cluster_indices = []
    diff_cluster_indices = []
    for j in range(len(kmeans_labels)):
        if kmeans_labels[j] == kmeans_prediction:
            same_cluster_indices.append(j)
        else:
            diff_cluster_indices.append(j)

    same_cluster_indices = np.array(same_cluster_indices)
    diff_cluster_indices = np.array(diff_cluster_indices)

    same_cluster_features = db_features[same_cluster_indices]
    same_cluster_distances = utils.euclidean_distance(query_feature, same_cluster_features)
    sorted_same_cluster_indices = same_cluster_indices[np.argsort(same_cluster_distances)]

    diff_cluster_features = db_features[diff_cluster_indices]
    diff_cluster_distances = utils.euclidean_distance(query_feature, diff_cluster_features)
    sorted_diff_cluster_indices = diff_cluster_indices[np.argsort(diff_cluster_distances)]

    sorted_index = np.concatenate((sorted_same_cluster_indices, sorted_diff_cluster_indices))

    # concat 2 results, which the same cluster results will be put to top
    sorted_labels = db_labels[sorted_index]

    tp = 0
    sum_precisions = 0
    for j in range(len(sorted_labels)):
        if sorted_labels[j] == query_label:
            tp += 1
            sum_precisions += tp / (j+1)

    return sum_precisions / total_relevents


def mean_precision_at_k(extractor, x, y, k):
    features = extractor.predict(x)
    return np.mean([
        precision_at_k_normal(features, y, i, k)
        for i in range(len(features))
    ])
    pass


def precision_at_k_normal(features, labels, index, k):
    query_feature = features[index]
    query_label = labels[index]

    db_features = np.delete(features, index, axis=0)
    db_labels = np.delete(labels, index, axis=0)

    distances = utils.euclidean_distance(query_feature, db_features)
    sorted_indices = np.argsort(distances)[:k]
    sorted_labels = db_labels[sorted_indices]

    n_true_labels = len(utils.where_equal(sorted_labels, query_label))
    return n_true_labels / k
