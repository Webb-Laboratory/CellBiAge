import Model
import Data_Preprocess
import tensorflow as tf
import numpy as np
from unittest import result
from sklearn.decomposition import PCA
from loss_visualization import visualize_plots


def main():
    train_X, test_X, train_y, test_y = Data_Preprocess.load_train_and_test("../data/adata_df_2k_grouped_with_cate_vars",
                                                                           "../data/adata_df_2k_grouped.csv",
                                                                           Data_Preprocess.onehot_encoding)

    ##################baseline MLP#####################
    baseline_model = Model.Baseline_MLP(feature_nums=[500, 100, 50, 25])

    baseline_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
                           loss=tf.keras.losses.BinaryCrossentropy(),
                           metrics=[tf.keras.metrics.BinaryAccuracy(),
                                    tf.keras.metrics.AUC()])
    history = baseline_model.fit(
        train_X, train_y,
        validation_data=(test_X, test_y),
        epochs=50,
        batch_size=100,
        verbose=2,
        # shuffle=True,  # add shuffle here
        # validation_split=0.2  # validation split for plots
    )

    # testing_result = baseline_model.evaluate(test_X, test_y, verbose=0)
    # visualize_plots(history)
    # print(dict(zip(baseline_model.metrics_names, testing_result)))

    ##################PCA#####################
    # pca = PCA(n_components = 100)
    # pca_train_X = pca.fit_transform(train_X)
    # pca_test_X = pca.transform(test_X)
    #
    # pca_model = Model.Baseline_MLP(feature_nums=[200, 50, 25])
    # pca_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    #                   loss=tf.keras.losses.BinaryCrossentropy(),
    #                   metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()])
    #
    # history = pca_model.fit(
    #     pca_train_X, train_y,
    #     validation_data=(pca_test_X, test_y),
    #     epochs=1000,
    #     batch_size=500,
    #     verbose=2,
    #     # shuffle = True,#add shuffle here
    #     # validation_split=0.2 #validation split for plots
    # )
    # pca_testing_result = pca_model.evaluate(pca_test_X, test_y, verbose=2)
    # visualize_plots(history)
    # print(dict(zip(pca_model.metrics_names, pca_testing_result)))
    # print(np.sum(test_y)/len(test_y))

    ##################XGBOOST#####################
    # df = pd.read_csv("../data/adata_df_2k_grouped.csv")
    # X, y = df.iloc[:, 1:-4], df["target"]
    # xgb = Model.Baseline_XGB("../data/adata_df_2k_grouped.csv", need_train=True)
    # xgb.train()
    # xgb.test(X, y)

    pass


if __name__ == '__main__':
    main()
