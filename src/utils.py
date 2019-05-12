import time

import math
from keras import backend as K


def euclidean_distance(a, b):
    return K.sqrt(K.sum(K.square(a - b)))


def get_triplet_index_all(labels):
    start = time.time()
    triplets = []

    for i in range(len(labels)):
        for j in range(len(labels)):
            if j == i or labels[j] != labels[i]:
                continue
            for k in range(len(labels)):
                if (i == k) or (j == k):
                    continue
                if labels[i] == labels[k]:
                    continue
                triplets.append([i, j, k])

    print('Time for finding all triplet index:', time.time() - start)
    return triplets


def get_triplet_index_hard(features, labels):
    start = time.time()
    triplets = []

    for i in range(len(labels)):
        max_pos_d = 0
        max_pos_idx = None
        min_neg_d = math.inf
        min_neg_idx = None

        for j in range(len(labels)):
            if j == i:  # ignore the same image
                continue

            distance = euclidean_distance(features[i], features[j])

            if labels[j] == labels[i]:  # positive
                if distance > max_pos_d:
                    max_pos_d = distance
                    max_pos_idx = j

            else:  # negative
                if distance < min_neg_d:
                    min_neg_d = distance
                    min_neg_idx = j

        triplets.append([i, max_pos_idx, min_neg_idx])

    print('Time for finding hard triplet index:', time.time() - start)
    return triplets
