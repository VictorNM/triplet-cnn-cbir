import numpy as np
from sklearn.cluster import KMeans
import time
from . import utils


def mAP(extractor, db, queries, mode='normal'):
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