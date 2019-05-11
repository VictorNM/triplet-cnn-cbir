from keras.callbacks import TensorBoard
from keras.utils import to_categorical
from keras.optimizers import SGD
import time
import csv
import os


def compile_classifier(classifier, optimizer_params):
    optimizer = _get_optimizer(optimizer_params)
    classifier.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return classifier


def train_classifier(classifier, dataset, training_params):
    x_train, y_train = dataset['x_train'], dataset['y_train']
    x_valid, y_valid = dataset['x_valid'], dataset['y_valid']

    classes = dataset['classes']
    metadata_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/metadata.tsv'))
    with open(metadata_file, 'wt') as f:
        tsv_writer = csv.writer(f, delimiter='\t')
        for label in y_valid:
            tsv_writer.writerow([classes[int(label)]])

    tensor_board = TensorBoard(
        batch_size=training_params['batch_size'],
        embeddings_freq=1,
        embeddings_layer_names=['fc2'],
        embeddings_data=x_valid,
        embeddings_metadata=metadata_file
    )

    start = time.time()
    train_history = classifier.fit(
        x=x_train,
        y=to_categorical(y_train),
        validation_data=(x_valid, to_categorical(y_valid)),
        callbacks=[tensor_board],
        **training_params
    )
    end = time.time()
    print('Total training time:', end - start)

    return classifier, train_history


def train_extractor():
    pass


def _get_optimizer(config_optimizer):
    if config_optimizer is None:
        return SGD()

    if config_optimizer['name'] == 'SGD':
        params = {
            k: config_optimizer[k]
            for k in config_optimizer.keys() - {'name'}
        }
        return SGD(**params)

    raise ValueError("Invalid optimizer configuration. Optimizer name should in {'SGD'}")
