import os
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
from six.moves import cPickle as pickle


_data_root = ''
STANDFORD_ONLINE_PRODUCTS = 'standford_online_products'
RAW = 'raw'
PROCESSED = 'processed'


def set_data_root(data_root):
    global _data_root

    if os.path.isabs(data_root):
        _data_root = data_root
    else:
        _data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), data_root))


def load(data_root='../../data', dataset_name=STANDFORD_ONLINE_PRODUCTS, dataset_type=RAW):
    set_data_root(data_root)
    print('Loading dataset: %s from folder: %s' % (dataset_name, _data_root))
    try:
        print('Try to read from pickle file...')
        dataset = _load_from_pickle_file(dataset_name, dataset_type)
        print('Load data succeed')
    except FileNotFoundError:
        print('Pickle file not found, try to read from directory')
        if dataset_name == STANDFORD_ONLINE_PRODUCTS:
            dataset = _load_standford_online_product(dataset_type)
        else:
            raise NotImplementedError

        # save pickle file for later
        print('Load data succeed')
        _save_pickle(dataset, dataset_name, dataset_type)

    return dataset


def subset(dataset, n_train=100, n_valid=20, n_test=20):
    return {
        'classes': dataset['classes'],
        'x_train': dataset['x_train'][:n_train],
        'y_train': dataset['y_train'][:n_train],
        'x_valid': dataset['x_valid'][:n_valid],
        'y_valid': dataset['y_valid'][:n_valid],
        'x_test': dataset['x_test'][:n_test],
        'y_test': dataset['y_test'][:n_test]
    }


def _load_from_pickle_file(dataset_name, dataset_type):
    pickle_file_name = dataset_name + '.pickle'
    pickle_file_path = os.path.join(_data_root, dataset_type, pickle_file_name)
    if os.path.exists(pickle_file_path):
        with open(pickle_file_path, 'rb') as f:
            dataset = pickle.load(f)
            return dataset

    raise FileNotFoundError


def _load_standford_online_product(datatype):
    data_path = os.path.join(_data_root, datatype, 'standford_online_products')
    train_data_path = os.path.join(data_path, 'train')
    test_data_path = os.path.join(data_path, 'test')

    train_generator = ImageDataGenerator(dtype=np.uint8).\
        flow_from_directory(train_data_path, class_mode='sparse')

    test_generator = ImageDataGenerator(dtype=np.uint8).\
        flow_from_directory(test_data_path, class_mode='sparse')

    x_train, y_train = _convert_generator_to_data(train_generator)
    x_test, y_test = _convert_generator_to_data(test_generator)

    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2)

    classes = [[] for i in range(train_generator.num_classes)]
    for (name, index) in train_generator.class_indices.items():
        classes[index] = name

    dataset = dict()
    dataset['classes'] = classes
    dataset['x_train'] = x_train
    dataset['y_train'] = y_train
    dataset['x_valid'] = x_valid
    dataset['y_valid'] = y_valid
    dataset['x_test'] = x_test
    dataset['y_test'] = y_test

    return dataset


def _save_pickle(data, dataset_name, dataset_type):
    pickle_file_name = dataset_name + '.pickle'
    pickle_file_path = os.path.join(_data_root, dataset_type, pickle_file_name)
    print('Saving dataset to %s' % pickle_file_path)
    try:
        f = open(pickle_file_path, 'wb')
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        print('Saved data success')
    except Exception as e:
        print('Unable to save data:', e)
        raise


def _convert_generator_to_data(generator):
    x = np.ndarray(shape=((generator.n,) + generator.image_shape), dtype=np.uint8)
    y = np.ndarray(shape=generator.n, dtype=np.uint8)
    for i in range(generator.__len__()):
        x_batch, y_batch = generator.next()
        start_index = i * generator.batch_size
        end_index = (i + 1) * generator.batch_size
        x[start_index:end_index] = x_batch
        y[start_index:end_index] = y_batch

    return x, y