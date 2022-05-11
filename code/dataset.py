import Data_Preprocess
from sklearn.decomposition import PCA


def PCA_dataset(dataset_func, n_components=100):
    train_X, test_X, train_y, test_y, dataset_name = dataset_func()
    pca = PCA(n_components=n_components)
    pca_train_X = pca.fit_transform(train_X)
    pca_test_X = pca.transform(test_X)

    return pca_train_X, pca_test_X, train_y, test_y, dataset_name + " plus PCA with {} components".format(n_components)


def one_hot_dataset():
    train_X, test_X, train_y, test_y = Data_Preprocess.load_train_and_test(
        "../data/adata_df_2k_grouped_one_hot_cate",
        "../data/adata_df_2k_grouped.csv",
        Data_Preprocess.onehot_encoding)

    return train_X, test_X, train_y, test_y, "One Hot Encoding Dataset"

def dca_one_hot_dataset():
    train_X, test_X, train_y, test_y = Data_Preprocess.load_train_and_test(
        "../data/dca_latent_one_hot_cate",
        "../data/dca_latent.csv",   #DCA needs
        Data_Preprocess.onehot_encoding)

    return train_X, test_X, train_y, test_y, "One Hot Encoding DCA Dataset"

def dca_one_hot_cate_binarized_std_dataset():
    train_X, test_X, train_y, test_y = Data_Preprocess.load_train_and_test(
        "../data/dca_latent_one_hot_cate",
        "../data/dca_latent.csv",   #DCA needs
        Data_Preprocess.onehot_encoding, Data_Preprocess.binarize_with_std_data)
    
    return train_X, test_X, train_y, test_y, "One Hot Encoding and binarized Dataset"

# To binarize `adata_df_preprocessed.csv`, which contains data that's already standardized (2000 cols)
def one_hot_cate_binarized_std_dataset():
    train_X, test_X, train_y, test_y = Data_Preprocess.load_train_and_test(
        "../data/adata_df_2k_grouped_one_hot_cate_binarized_std",
        "../data/adata_df_2k_grouped.csv",
        Data_Preprocess.onehot_encoding, Data_Preprocess.binarize_with_std_data)

    return train_X, test_X, train_y, test_y, "One Hot Encoding and binarized Dataset"


# For `adata_df_origin_1888`, which contains data that's not yet standardized (1888 cols)
def one_hot_binarized_origin_dataset():
    train_X, test_X, train_y, test_y = Data_Preprocess.load_train_and_test(
        "../data/adata_df_origin_1888_one_hot_cate_binarized_origin",
        "../data/adata_df_origin_1888.csv",
        Data_Preprocess.onehot_encoding, Data_Preprocess.binarize_with_origin_data)

    return train_X, test_X, train_y, test_y, "One Hot Encoding and binarized Dataset"


def without_cate_binarized_dataset():
    train_X, test_X, train_y, test_y = Data_Preprocess.load_train_and_test(
        "../data/adata_df_2k_grouped_without_cate_binarized",
        "../data/adata_df_2k_grouped.csv",
        Data_Preprocess.exclude_cate, Data_Preprocess.binarize_with_std_data)

    return train_X, test_X, train_y, test_y, "Category excluded and binarized Dataset"


def without_cate_dataset():
    train_X, test_X, train_y, test_y = Data_Preprocess.load_train_and_test(
        "../data/adata_df_2k_grouped_without_cate",
        "../data/adata_df_2k_grouped.csv",
        Data_Preprocess.exclude_cate)

    return train_X, test_X, train_y, test_y, "Category excluded Dataset"
