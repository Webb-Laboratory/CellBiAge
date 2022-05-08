from unittest import result
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

    ##################baseline MLP#####################
    # baseline_model = Model.Baseline_MLP(feature_nums=[500, 100, 50, 25])
    
    # baseline_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
    #                        loss=tf.keras.losses.BinaryCrossentropy(),
    #                        metrics=[tf.keras.metrics.BinaryAccuracy(),
    #                                 tf.keras.metrics.AUC()])
    # history = baseline_model.fit(
    #     train_X, train_y,
    #     epochs=50, 
    #     batch_size=100,
    #     verbose=2,
    #     shuffle = True,#add shuffle here 
    #     validation_split=0.2 #validation split for plots
    # )

    
    # testing_result = baseline_model.evaluate(test_X, test_y, verbose=0) 
    # visualize_plots(history)
    # print(dict(zip(baseline_model.metrics_names, testing_result)))
   
    
    ##################PCA#####################
    pca = PCA(n_components = 100)
    pca_train_X = pca.fit_transform(train_X)
    pca_test_X = pca.transform(test_X)
    
    pca_model = Model.Baseline_MLP()
    pca_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()])
    
    history = pca_model.fit(
        pca_train_X, train_y,
        epochs=450, 
        batch_size=10000,
        verbose=2,
        shuffle = True,#add shuffle here 
        validation_split=0.2 #validation split for plots
    )
    pca_testing_result = pca_model.evaluate(pca_test_X, test_y, verbose=2)
    visualize_plots(history)
    print(dict(zip(pca_model.metrics_names, pca_testing_result)))
    print(np.sum(test_y)/len(test_y))

    ##################XGBOOST#####################
    # df = pd.read_csv("../data/adata_df_2k_grouped.csv")
    # X, y = df.iloc[:, 1:-4], df["target"]
    # xgb = Model.Baseline_XGB("../data/adata_df_2k_grouped.csv", need_train=True)
    # xgb.train()
    # xgb.test(X, y)

def visualize_plots(history): 
    """
    Uses Matplotlib to visualize the losses and metrics of our model.
    :param history: dictionary of losses and metrics for train and validation data
    :return: doesn't return anything, three plots should pop-up 
    """
    
    #print(history.history.keys()) # list all data in history

    # summarize history for accuracy
    plt.plot(history.history['binary_accuracy'])
    plt.plot(history.history['val_binary_accuracy'])
    plt.title('model binary_accuracy')
    plt.ylabel('binary_accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
    
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    # summarize history for AUC
    plt.plot(history.history['auc'])
    plt.plot(history.history['val_auc'])
    plt.title('model auc')
    plt.ylabel('auc')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

if __name__ == '__main__':
    main()






