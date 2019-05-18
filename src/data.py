import numpy as np


def test():
    print('hello')


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

# def save_pickle(dataset, dataset_name, dataset_type):
#     pickle_file_name = dataset_name + '.pickle'
#     pickle_file_path = os.path.join(_data_root, dataset_type, pickle_file_name)
#     print('Saving dataset to %s' % pickle_file_path)
#     try:
#         f = open(pickle_file_path, 'wb')
#         pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
#         f.close()
#         print('Saved dataset success')
#     except Exception as e:
#         print('Unable to save dataset:', e)
#         raise