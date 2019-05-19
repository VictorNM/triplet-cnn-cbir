import os
import time
from csv import writer

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from requests import get


def get_tensorboard_url(logs_dir):
    # download and unzio ngrok
    if not os.path.exists('ngrok-stable-linux-amd64.zip'):
        os.system(
            'wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip')
    if not os.path.exists('ngrok'):
        os.system('unzip ngrok-stable-linux-amd64.zip')

    # init host
    os.system(
        'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'.format(logs_dir))
    os.system('./ngrok http 6006 &')

    time.sleep(2) # wait for host initialize

    res = get('http://localhost:4040/api/tunnels')

    return res.json()['tunnels'][0]['public_url']


def get_tensorboard(logs_dir, datetime_train, x, y, classes, params):
    log_dir = os.path.join(logs_dir, datetime_train)
    metadata_path = os.path.join(log_dir, 'metadata.tsv')

   # create metadata file
    if not os.path.exists(os.path.dirname(metadata_path)):
        os.makedirs(os.path.dirname(metadata_path))

    with open(metadata_path, 'wt') as f:
        tsv_writer = writer(f, delimiter='\t')
        for label in y:
            tsv_writer.writerow([classes[int(label)]])

        f.close()

    return TensorBoard(
        log_dir=log_dir,
        embeddings_freq=1,
        embeddings_layer_names=['fc2'],
        embeddings_data=x,
        embeddings_metadata='metadata.tsv',
        **params
    )


def get_model_checkpoint(models_dir, datetime_train, params):
    model_file_path = os.path.join(models_dir, datetime_train + '.h5')
    if not os.path.exists(os.path.dirname(model_file_path)):
        os.makedirs(os.path.dirname(model_file_path))

    return ModelCheckpoint(
        filepath=model_file_path,
        **params
    )

def get_early_stopping(params):
    return EarlyStopping(**params)
