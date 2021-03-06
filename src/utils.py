import os
import time
from csv import writer
from datetime import datetime
from math import inf

import numpy as np
import requests
from keras import backend as K
from pytz import country_timezones, timezone
import pickle


def euclidean_distance_keras(a, b):
    return K.sqrt(K.sum(K.square(a - b), axis=-1))


def euclidean_distance(a, b):
    return np.sqrt(np.sum(np.square(a - b), axis=-1))


def where_equal(arr, val):
    # return index of values in arr that equal to val
    # return type: 1-D numpy array
    return np.array(np.asanyarray(arr == val).nonzero()).flatten()


def get_triplet_index_all(labels):
    start = time.time()
    triplets = []

    for i in range(len(labels)):
        for j in range(len(labels)):
            if j == i or labels[j] != labels[i]:
                continue
            for k in range(len(labels)):
                if labels[i] == labels[k]:
                    continue
                triplets.append([i, j, k])

    print('Found %d triplet index in %s:' % (len(triplets), time.time() - start))
    return triplets


def get_triplet_index_hard(features, labels):
    start = time.time()
    triplets = []

    for i in range(len(labels)):
        max_pos_d = 0
        max_pos_idx = None
        min_neg_d = inf
        min_neg_idx = None

        for j in range(len(labels)):
            if j == i:  # ignore the same image
                continue

            distance = euclidean_distance_keras(features[i], features[j])

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

def get_datetime_train():
    tz = timezone(country_timezones('vn')[0])
    return datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')


def save_pickle(data, file_path):
    pickle_file_path = file_path + '.pickle'
    try:
        f = open(pickle_file_path, 'wb')
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to save:', e)
        raise


def load_pickle(file_path):
    pickle_file_path = file_path + '.pickle'
    try:
        f = open(pickle_file_path, 'rb')
        res = pickle.load(f)
        f.close()
        return res
    except Exception as e:
        print('Unable to load:', e)
        raise