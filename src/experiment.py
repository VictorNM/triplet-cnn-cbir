import time

import numpy as np
from sklearn.cluster import KMeans

from . import utils


def mAP(extractor, db, queries):
    db_images, db_labels = db
    db_features = extractor.predict(db_images)

    query_images, query_labels = queries
    query_features = extractor.predict(query_images)
    
    return np.mean(
        [AP(
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

    sum_ap = 0

    for i in range(len(query_labels)):
        query_feature = query_features[i]
        query_label = query_labels[i]
        kmean_prediction = kmean_predictions[i]

        total_relevents = len(utils.where_equal(db_labels, query_label))
        if total_relevents == 0:
            continue

        same_cluster_indices = []
        diff_cluster_indices = []
        for j in range(len(kmeans.labels_)):
            if kmeans.labels_[j] == kmean_prediction:
                same_cluster_indices.append(j)
            else:
                diff_cluster_indices.append(j)
                
        same_cluster_indices = np.array(same_cluster_indices)
        diff_cluster_indices = np.array(diff_cluster_indices)

        same_cluster_features = db_features[same_cluster_indices]
        same_cluster_labels = db_labels[same_cluster_indices]
        same_cluster_distances = utils.euclidean_distance(query_feature, same_cluster_features)

        sorted_same_cluster_indices = np.argsort(same_cluster_distances)
        sorted_same_cluster_labels = db_labels[same_cluster_indices[sorted_same_cluster_indices]]

        diff_cluster_features = db_features[diff_cluster_indices]
        diff_cluster_labels = db_labels[diff_cluster_indices]
        diff_cluster_distances = utils.euclidean_distance(query_feature, diff_cluster_features)

        sorted_diff_cluster_indices = np.argsort(diff_cluster_distances)
        sorted_diff_cluster_labels = db_labels[diff_cluster_indices[sorted_diff_cluster_indices]]

        sorted_labels = np.concatenate(sorted_same_cluster_labels, sorted_diff_cluster_labels)

        tp = 0
        sum_precisions = 0
        for j in range(len(sorted_labels)):
            if sorted_labels[j] == query_label:
                tp += 1
                sum_precisions += tp / (j+1)

        sum_ap += (sum_precisions / tp)

    return sum_ap / len(query_labels)


def AP(query_feature, query_label, db_features, db_labels):
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

    return sum_precisions / tp
