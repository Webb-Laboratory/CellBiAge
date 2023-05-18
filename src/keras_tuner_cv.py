import tensorflow as tf
import keras_tuner as kt
from sklearn.utils import shuffle
import pandas as pd
import numpy as np

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
        early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='AUPRC', patience=10)
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


