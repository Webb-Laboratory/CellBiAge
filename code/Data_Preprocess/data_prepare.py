import numpy as np
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split

from .preprocess_methods import onehot_encoding


def split_train_test(data, label):
    train_X, test_X, train_y, test_y = train_test_split(data, label,
                                                        test_size=0.2, random_state=13)

    train_X = np.array(train_X).astype(dtype=np.float32)
    test_X = np.array(test_X).astype(dtype=np.float32)
    train_y = np.array(train_y).astype(dtype=np.int32)
    test_y = np.array(test_y).astype(dtype=np.int32)

    return train_X, test_X, train_y, test_y


def dump_data(dir, train_X, test_X, train_y, test_y):
    os.makedirs(dir, exist_ok=True)

    with open(os.path.join(dir, 'train.pickle'), 'wb') as handle:
        pickle.dump((train_X, train_y), handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(dir, 'test.pickle'), 'wb') as handle:
        pickle.dump((test_X, test_y), handle, protocol=pickle.HIGHEST_PROTOCOL)


def prepare_data(path, target_dir=None, *preprocesses):
    parent_dir, file = os.path.split(path)
    file_name = file.split(".")[0]
    if not target_dir:
        target_dir = os.path.join(parent_dir, file_name)

    input = (pd.read_csv(path)).iloc[:, 1:]  # exclude index col
    label = input['target']
    
    #data = input.drop(columns=["target", "animal"])
    data = input.drop(columns=["target"])

    for one_preprocess in preprocesses:
        data = one_preprocess(data)

    split_data = split_train_test(data, label)
    dump_data(target_dir, *split_data)


def main():
    #prepare_data("../../data/adata_df_2k_grouped.csv", onehot_encoding)
    prepare_data("../data/dca_latent.csv", onehot_encoding)

if __name__ == '__main__':
    main()
