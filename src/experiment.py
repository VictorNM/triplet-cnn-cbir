import time

import numpy as np
from sklearn.cluster import KMeans

from . import utils


def mAP_normal(extractor, db, queries):
    db_images, db_labels = db
    db_features = extractor.predict(db_images)

    query_images, query_labels = queries
    query_features = extractor.predict(query_images)

    return np.mean(
        [AP_normal(
            query_feature=query_features[i],
            query_label=query_labels[i],
            db_features=db_features,
            db_labels=db_labels
        )
            for i in range(len(query_features))]
    )


def mAP_kmeans(extractor, db, queries):
    db_images, db_labels = db
    db_features = extractor.predict(db_images)

    query_images, query_labels = queries
    query_features = extractor.predict(query_images)

    num_classes = len(set(db_labels))
    kmeans = KMeans(num_classes)
    kmeans.fit(db_features)

    kmean_predictions = kmeans.predict(query_features)

    num_queries = len(query_labels)

    return np.mean([
        AP_kmeans(
            query_feature=query_features[i],
            query_label=query_labels[i],
            kmean_prediction=kmean_predictions[i],
            db_features=db_features,
            db_labels=db_labels,
            kmean_labels=kmeans.labels_
        )
        for i in range(num_queries)
    ])


def AP_normal(query_feature, query_label, db_features, db_labels):
    total_relevents = len(utils.where_equal(db_labels, query_label))

    # if db doesn't have relevants image, return 0
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

    if tp == 0:
        return 0

    return sum_precisions / total_relevents


def AP_kmeans(query_feature, query_label, kmean_prediction, db_features, db_labels, kmean_labels):
    total_relevents = len(utils.where_equal(db_labels, query_label))

    # if db doesn't have relevants image, return 0
    if total_relevents == 0:
        return 0

    # all features in same cluster will be sorted and put to top
    # other features will also be sorted and put after

    same_cluster_indices = []
    diff_cluster_indices = []
    for j in range(len(kmean_labels)):
        if kmean_labels[j] == kmean_prediction:
            same_cluster_indices.append(j)
        else:
            diff_cluster_indices.append(j)

    same_cluster_indices = np.array(same_cluster_indices)
    diff_cluster_indices = np.array(diff_cluster_indices)

    same_cluster_features = db_features[same_cluster_indices]
    same_cluster_distances = utils.euclidean_distance(
        query_feature, same_cluster_features)

    sorted_same_cluster_indices = np.argsort(same_cluster_distances)
    sorted_same_cluster_labels = db_labels[same_cluster_indices[sorted_same_cluster_indices]]

    diff_cluster_features = db_features[diff_cluster_indices]
    diff_cluster_distances = utils.euclidean_distance(
        query_feature, diff_cluster_features)

    sorted_diff_cluster_indices = np.argsort(diff_cluster_distances)
    sorted_diff_cluster_labels = db_labels[diff_cluster_indices[sorted_diff_cluster_indices]]

    # concat 2 results, which the same cluster results will be put to top
    sorted_labels = np.concatenate(
        (sorted_same_cluster_labels, sorted_diff_cluster_labels))

    tp = 0
    sum_precisions = 0
    for j in range(len(sorted_labels)):
        if sorted_labels[j] == query_label:
            tp += 1
            sum_precisions += tp / (j+1)

    return sum_precisions / total_relevents



def mean_precision_at_k():
    pass


def precision_at_k():
    pass