import os
import pickle
from .data_prepare import prepare_data


def load_data_from_pickle(path):
    with open(path, 'rb') as handle:
        data, label = pickle.load(handle)

    return data, label


def load_train_and_test(parent_dir, data_path=None, *preprocesses):
    if not data_path:
        data_path = parent_dir + ".csv"

    if not os.path.exists(parent_dir):
        prepare_data(data_path, parent_dir, *preprocesses)

    train_X, train_y = load_data_from_pickle(os.path.join(parent_dir, "train.pickle"))
    test_X, test_y = load_data_from_pickle(os.path.join(parent_dir, "test.pickle"))

    return train_X, test_X, train_y, test_y
