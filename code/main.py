import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from xgboost import XGBClassifier as xgb


class Baseline_MLP(tf.keras.Model):

    def __init__(self):

        super().__init__()
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense((64), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense((32), activation='relu'),
            tf.keras.layers.Dense((16), activation='relu'),
            tf.keras.layers.Dense((1), activation='sigmoid')
        ])

    def call(self, inputs, training=True):

        prob = self.mlp(inputs)

        return prob

def main():

    df = pd.read_csv("../data/adata_df_2k_grouped.csv")

    train_X, test_X, train_y, test_y = train_test_split(df.iloc[:, 1:-4], df["target"],\
                                                        test_size=0.2, random_state=13)

    train_X = np.array(train_X).astype(dtype=np.float32)
    test_X = np.array(test_X).astype(dtype=np.float32)
    train_y = np.array(train_y).astype(dtype=np.int32)
    test_y = np.array(test_y).astype(dtype=np.int32)

    #baseline MLP
    baseline_model = Baseline_MLP()

    baseline_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                           loss=tf.keras.losses.BinaryCrossentropy(),
                           metrics=[tf.keras.metrics.BinaryAccuracy(),
                                    tf.keras.metrics.AUC()])
    baseline_model.fit(
        train_X, train_y,
        epochs=20,
        batch_size=10000,
        verbose=2
    )
    testing_result=baseline_model.evaluate(test_X, test_y, verbose=2)
    print(dict(zip(baseline_model.metrics_names, testing_result)))
    print(np.sum(test_y)/len(test_y))

    #PCA
    pca = PCA(n_components = 100)
    pca_train_X = pca.fit_transform(train_X)
    pca_test_X = pca.transform(test_X)

    pca_model = Baseline_MLP()
    pca_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                           loss=tf.keras.losses.BinaryCrossentropy(),
                           metrics=[tf.keras.metrics.BinaryAccuracy(),
                                    tf.keras.metrics.AUC()])
    pca_model.fit(
        pca_train_X, train_y,
        epochs=300,
        batch_size=10000,
        verbose=2
    )
    pca_testing_result=pca_model.evaluate(pca_test_X, test_y, verbose=2)
    print(dict(zip(pca_model.metrics_names, pca_testing_result)))
    print(np.sum(test_y)/len(test_y))

    #try xgboost






if __name__ == '__main__':
    main()






