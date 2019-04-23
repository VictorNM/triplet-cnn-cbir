def train_classifier(classifier, x, y, config):
    train_history = classifier.fit(
        x=x,
        y=y,
        batch_size=config.batch_size,
        epochs=config.epochs,
        validation_split=config.validation_split
    )
    return classifier, train_history


def train_extractor():
    pass
