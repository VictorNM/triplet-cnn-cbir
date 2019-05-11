from src.data import data_provider, data_processor
from src.model import model_provider
from src import training


def free_mem():
    for name in dir():
        if not name.startswith('_'):
            del globals()[name]


if __name__ == '__main__':
    # Prepare data
    dataset_raw = data_provider.load(
        data_root='/home/victor/Learning/bku/dissertation/implementation/data'
    )
    dataset_raw = data_provider.subset(dataset_raw) # get smaller dataset for testing
    input_shape = (224, 224, 3)
    dataset_inter = data_processor.normalize(dataset_raw, input_shape)
    dataset_final = data_processor.augment(dataset_inter, {})

    # Train CNN
    cnn_classifier = model_provider.build_cnn_classifier(
        model_name='vgg16',
        input_shape=input_shape,
        num_classes=len(dataset_final['classes'])
    )

    optimizer_params = {
        "name": "SGD",
        "lr": 0.02,
        "momentum": 0.9
    }

    cnn_classifier = training.compile_classifier(cnn_classifier, optimizer_params)

    # train model
    training_params = {
        "batch_size": 32,
        "epochs": 1
    }

    cnn_classifier, history = training.train_classifier(cnn_classifier, dataset_final, training_params)

    free_mem()
