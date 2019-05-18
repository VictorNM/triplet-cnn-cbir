import numpy as np

def test():
    return 'hello'


def convert_generator_to_data(generator):
    generator.reset()

    x_shape = (generator.n,) + generator.image_shape
    x = np.ndarray(shape=x_shape, dtype=generator.dtype)

    y_shape = generator.n
    y = np.ndarray(shape=y_shape, dtype=generator.dtype)

    for i in range(generator.__len__()):
        x_batch, y_batch = generator.next()
        start_index = i * generator.batch_size
        end_index = (i + 1) * generator.batch_size
        x[start_index:end_index] = x_batch
        y[start_index:end_index] = np.argmax(y_batch, axis=1)

    return x, y