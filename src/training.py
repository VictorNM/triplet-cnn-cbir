import keras


def train_classifier(classifier, x, y, config):
    opt = _get_optimizer(config.optimizer)
    classifier.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    train_history = classifier.fit(
        x=x,
        y=y,
        batch_size=config.batch_size,
        epochs=config.epochs,
        validation_split=config.validation_split,
        callbacks=[keras.callbacks.TensorBoard()]
    )
    return classifier, train_history


def train_extractor():
    pass


def _get_optimizer(config_optimizer):
    if config_optimizer is None:
        return keras.optimizers.SGD()

    if config_optimizer['name'] == 'SGD':
        params = {
            k: config_optimizer[k]
            for k in config_optimizer.keys() - {'name'}
        }
        return keras.optimizers.SGD(**params)

    raise ValueError("Invalid optimizer configuration. Optimizer name should in {'SGD'}")
