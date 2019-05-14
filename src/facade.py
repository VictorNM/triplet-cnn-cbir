from src.data import data_provider, data_processor
from src.model import model_provider
from src import training, experiment, kmeandatabase
import gc


def free_mem():
    for name in dir():
        if not name.startswith('_'):
            del globals()[name]

    gc.collect()


if __name__ == '__main__':
    # Prepare data
    dataset_raw = data_provider.load(
        data_root='/home/victor/Learning/bku/dissertation/implementation/data'
    )
    dataset_raw = data_provider.subset(dataset_raw, 10, 5, 5) # get smaller dataset for testing
    gc.collect()

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

    cnn_extractor = model_provider.build_cnn_extractor(cnn_classifier)
    dataset_final = data_provider.add_triplet_index(dataset_final)

    evaluate_params = {
        "top_k": 1
    }
    scores = experiment.evaluate_extractor(cnn_extractor, dataset_final, mode='valid', evaluate_params=evaluate_params)

    db = kmeandatabase.KmeanDatabase(cnn_extractor)
    db.create_database(dataset_final['x_train'], dataset_final['y_train'], dataset_final['classes'])
    db.query(dataset_final['x_valid'][0])

    free_mem()
