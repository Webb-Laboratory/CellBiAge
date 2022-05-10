import Model
import Data_Preprocess
import tensorflow as tf
import numpy as np
from unittest import result
from sklearn.decomposition import PCA
from loss_visualization import visualize_plots


def main():
    # train_X, test_X, train_y, test_y = Data_Preprocess.load_train_and_test("../data/adata_df_2k_grouped_one_hot_cate_binarized",
    #                                                                        "../data/adata_df_2k_grouped.csv",
    #                                                                        Data_Preprocess.onehot_encoding, Data_Preprocess.binarize)

    train_X, test_X, train_y, test_y = Data_Preprocess.load_train_and_test("./data/adata_df_2k_grouped_without_cate_binarized",
                                                                           "./data/adata_df_2k_grouped.csv",
                                                                           Data_Preprocess.exclude_cate, Data_Preprocess.onehot_encoding)

    ##################baseline MLP#####################
    # baseline_model = Model.Baseline_MLP(feature_nums=[160, 50, 25], dropout_rate=0.25)
    
    # baseline_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
    #                        loss=tf.keras.losses.BinaryCrossentropy(),
    #                        metrics=[tf.keras.metrics.BinaryAccuracy(),
    #                                 tf.keras.metrics.AUC()])
    # history = baseline_model.fit(
    #     train_X, train_y,
    #     validation_data=(test_X, test_y),
    #     epochs=50,
    #     batch_size=100,
    #     verbose=2
    # )
    
    # testing_result = baseline_model.evaluate(test_X, test_y, verbose=0)
    # visualize_plots(history)
    # print(dict(zip(baseline_model.metrics_names, testing_result)))

    ##################PCA#####################
    # pca = PCA(n_components=100)
    # pca_train_X = pca.fit_transform(train_X)
    # pca_test_X = pca.transform(test_X)

    # pca_model = Model.Baseline_MLP(feature_nums=[64, 32, 16, 8])
    # pca_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    #                   loss=tf.keras.losses.BinaryCrossentropy(),
    #                   metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()])

    # history = pca_model.fit(
    #     pca_train_X, train_y,
    #     validation_data=(pca_test_X, test_y),
    #     epochs=100,
    #     batch_size=1000,
    #     verbose=2,
    # )
    # pca_testing_result = pca_model.evaluate(pca_test_X, test_y, verbose=2)
    # visualize_plots(history)
    # print(dict(zip(pca_model.metrics_names, pca_testing_result)))
    # print(np.sum(test_y)/len(test_y))

    ##################XGBOOST#####################
    # xgb = Model.Baseline_XGB(need_train=True)  # once trained, set `need_train` to false to let model load local parameters
    # xgb.train(train_X, train_y)
    # xgb.test(test_X, test_y)

    ##################Autoencoder#####################
    autoencoder_model = Model.Dense_AE(100)
    autoencoder_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()])
    
    ae_train_X = autoencoder_model.encode(train_X)
    ae_test_X = autoencoder_model.encode(test_X)
    
    ae_model = Model.Baseline_MLP(feature_nums=[64, 32, 16, 8])
    ae_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()])
    
    history = ae_model.fit(
        ae_train_X, train_y,
        validation_data=(ae_test_X, test_y),
        epochs=20,
        batch_size=1000,
        verbose=2,
    )
    
    testing_result = ae_model.evaluate(ae_test_X, test_y, verbose=0)
    visualize_plots(history)
    print(dict(zip(ae_model.metrics_names, testing_result)))

if __name__ == '__main__':
    main()
