import Model
from dataset import *
from train import *


def main():
    # define save directory
    save_dir = "../results"

    # define dataset
    train_X, test_X, train_y, test_y, dataset_name = one_hot_binarized_origin_dataset()
    # train_X, test_X, train_y, test_y, dataset_name = PCA_dataset(without_cate_binarized_dataset, n_components=100)    # if want PCA

    # define train parameters
    parameters = train_params(
        learning_rate=0.001,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()],
        epochs=100,
        batch_size=10000
    )

    # define your model
    model = Model.Baseline_MLP(feature_nums=[100, 50, 10], dropout_rate=0.25)
    model_name = "MLP_layer_100_50_10_lr_001_with_cate_bin1888"   # !!!!!! Be sure to change it every time, or your local record will be overwritten

    # Start Training, after trainning, all records will be dumped to the `save_dir/model_name` dir
    train(os.path.join(save_dir, model_name), model, parameters, train_X, train_y, test_X, test_y, dataset_name)


if __name__ == '__main__':
    main()
