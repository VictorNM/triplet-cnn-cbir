import numpy as np


def test():
    print('hello')


def convert_generator_to_data(generator, batchs=None):
    generator.reset()

    if batchs is None or batchs > generator.__len__():
        batchs = generator.__len__()

    num_samples = min(batchs * generator.batch_size, generator.n)
    x_shape = (num_samples, ) + generator.image_shape
    x = np.empty(shape=x_shape)

    y_shape = (num_samples, )
    y = np.empty(shape=y_shape)

    for i in range(batchs):
        x_batch, y_batch = generator.__getitem__(i)
        start_index = i * generator.batch_size
        end_index = start_index + len(x_batch)
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