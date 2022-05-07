import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.decomposition import PCA
#from xgboost import XGBClassifier as xgb
import Model
import Data_Preprocess
from matplotlib import pyplot as plt


def main():
    train_X, test_X, train_y, test_y = Data_Preprocess.load_train_and_test("../data/adata_df_2k_grouped")

    #baseline MLP
    baseline_model = Model.Baseline_MLP(feature_nums=[500, 100, 50, 25])
    
    baseline_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
                           loss=tf.keras.losses.BinaryCrossentropy(),
                           metrics=[tf.keras.metrics.BinaryAccuracy(),
                                    tf.keras.metrics.AUC()])
    baseline_model.fit(
        train_X, train_y,
        epochs=50,
        batch_size=1000,
        verbose=2
    )
    testing_result = baseline_model.evaluate(test_X, test_y, verbose=2)
    print(dict(zip(baseline_model.metrics_names, testing_result)))

    # #PCA
    # pca = PCA(n_components = 100)
    # pca_train_X = pca.fit_transform(train_X)
    # pca_test_X = pca.transform(test_X)
    #
    # pca_model = Model.Baseline_MLP()
    # pca_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    #                   loss=tf.keras.losses.BinaryCrossentropy(),
    #                   metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()])
    #
    # pca_model.fit(
    #     pca_train_X, train_y,
    #     epochs=450,
    #     batch_size=10000,
    #     verbose=2
    # )
    # pca_testing_result = pca_model.evaluate(pca_test_X, test_y, verbose=2)
    # print(dict(zip(pca_model.metrics_names, pca_testing_result)))
    # print(np.sum(test_y)/len(test_y))

    # xgboost
    # df = pd.read_csv("../data/adata_df_2k_grouped.csv")
    # X, y = df.iloc[:, 1:-4], df["target"]
    # xgb = Model.Baseline_XGB("../data/adata_df_2k_grouped.csv", need_train=True)
    # xgb.train()
    # xgb.test(X, y)

if __name__ == '__main__':
    main()

def visualize_loss(losses): 
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list 
    field 
    NOTE: DO NOT EDIT
    :return: doesn't return anything, a plot should pop-up 
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()




