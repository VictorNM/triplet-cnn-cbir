import numpy as np


def test():
    print('hello')


def convert_generator_to_data(generator, batchs=None):
    generator.reset()

    if batchs is None:
        batchs = generator.__len__()

    x = []
    y = []

    for i in range(batchs):
        x_batch, y_batch = generator.__getitem__(i)
        x.append(x_batch)
        y.append(np.argmax(y_batch, axis=1))

    return np.array(x), np.array(y)

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