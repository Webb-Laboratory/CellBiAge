import tensorflow as tf
import keras_tuner as kt
from sklearn.utils import shuffle
from .binarization import binarize_data
import pandas as pd
import numpy as np

def data_prep(input_test, input_train, cell_type, binarization=True):
    df_test = pd.read_csv(input_test, index_col=0)
    df_train = pd.read_csv(input_train, index_col=0)

    if (cell_type=='All'):
        test_idx = df_test.index
        train_idx = df_train.index        
        
    elif (cell_type=='Non-neuronal'):
        test_idx= df_test.loc[df_test.major_group!='Neuron'].index      
        train_idx= df_train.loc[df_train.major_group!='Neuron'].index                          
                             
    else: 
        test_idx = df_test.loc[df_test.major_group==cell_type].index
        train_idx = df_train.loc[df_train.major_group==cell_type].index

    assert len(test_idx)>0, "This cell type doesn't exit in the test set. \n Or you may have a typo :("
    assert len(train_idx)>0, "This cell type doesn't exit in the training set. \n Or you may have a typo :("

    test_X = df_test.loc[test_idx].iloc[:,:-3]
    test_y = df_test.target
    test_y = test_y.loc[test_idx]
    test_X, test_y = shuffle(test_X, test_y, random_state=42)
    

    df_train = df_train.loc[train_idx]
    train = df_train.reset_index()
    index_13, index_24 = train.loc[(train['animal'] == 7)|(train['animal'] == 3),].index, train.loc[(train['animal'] == 8)|(train['animal'] == 4),].index
    
    index_14, index_23 = train.loc[(train['animal'] == 7) | (train['animal'] == 4),].index, train.loc[(train['animal'] == 8)|(train['animal'] == 3),].index
    
    custom_cv = [(index_13, index_24), 
                 (index_14, index_23),
                 (index_23, index_14),
                 (index_24, index_13)]

    train_X = train.iloc[:,1:-3]
    train_y = train['target']
    
    if binarization==True:
        test_X = binarize_data(test_X)
        train_X = binarize_data(train_X)
    
    print('Finished data prepration')
    return train_X, train_y, test_X, test_y, custom_cv

def build_model(hp):
    n_hidden = hp.Int("n_hidden", min_value=0, max_value=8, default=2)
    n_neurons = hp.Int("n_neurons", min_value=32, max_value=1024, step = 32)
    learning_rate = hp.Float("learning_rate", min_value=1e-8, max_value=1e-2,
                             sampling="log")
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model = tf.keras.Sequential()

    for _ in range(n_hidden):
        model.add(tf.keras.layers.Dense(n_neurons, activation="relu",
                                        kernel_initializer=tf.keras.initializers.HeNormal()))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid", 
                                        kernel_initializer=tf.keras.initializers.HeNormal()))
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), 
                      optimizer=optimizer,
                      metrics=tf.keras.metrics.AUC(num_thresholds=10000, name='AUPRC', curve='PR'))
    return model

class CVTuner(kt.engine.tuner.Tuner):
    def run_trial(self, trial, train_X, train_y, custom_cv):
        val_losses = []
        AUPRC = []
        early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='AUPRC', patience=20)
        for i in range(len(custom_cv)):
            print('combination', i)
            X_train = train_X.iloc[custom_cv[i][0]]
            y_train = train_y[custom_cv[i][0]]
            X_train, y_train = shuffle(X_train, y_train, random_state=42)

            X_val = train_X.iloc[custom_cv[i][1]]
            y_val = train_y[custom_cv[i][1]]
            X_val, y_val = shuffle(X_val, y_val, random_state=42)

            print('train:', y_train.value_counts(), 'validation:',y_val.value_counts())
            
            model = self.hypermodel.build(trial.hyperparameters)
            model.fit(X_train, y_train, batch_size=32, epochs=50, callbacks=[early_stopping_cb])
                          
            val_result = model.evaluate(X_val, y_val, verbose=0)
            
            print(val_result)
            
            val_losses.append(val_result[0])
            AUPRC.append(val_result[1])

        print(np.mean(val_losses), np.mean(AUPRC))
        self.oracle.update_trial(trial.trial_id, {'val_AUPCR': np.mean(AUPRC)})


