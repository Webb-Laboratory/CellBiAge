import tensorflow as tf
import random
import os
import pickle
from functools import partial
from sklearn.utils import shuffle


class Baseline_MLP(tf.keras.Model):

    def __init__(self, feature_nums=None, dropout_rate=0, initializer=tf.keras.initializers.HeNormal(), **kwargs):
        super().__init__(**kwargs)
        if feature_nums is None:
            feature_nums = [64, 32, 16]

        self.initializer = initializer

        self.mlp = tf.keras.Sequential()

        for i in feature_nums:
            self.mlp.add(tf.keras.layers.Dense(i, activation='relu', kernel_initializer=self.initializer))
                                                                     # kernel_regularizer=tf.keras.regularizers.l2(0.001)))
            if dropout_rate > 0:
                self.mlp.add(tf.keras.layers.Dropout(dropout_rate))
       
        #self.mlp = tf.keras.Sequential([tf.keras.layers.Dense(i, activation='relu', kernel_initializer=self.initializer) for i in feature_nums] )                              
        #self.mlp.add(tf.keras.layers.Dense(i, activation=partial(tf.nn.leaky_relu, alpha=0.01), kernel_initializer=self.initializer))                    
        self.pred_head = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=self.initializer)


    def call(self, inputs):
        x = self.mlp(inputs)
        prob = self.pred_head(x)
        return prob


class Train_Params:
    def __init__(self, learning_rate, loss, metrics, epochs, batch_size):
        self.learning_rate = learning_rate
        self.loss = loss
        self.metrics = metrics
        self.epochs = epochs
        self.batch_size = batch_size

    def __repr__(self):
        parameters = {
            "learning_rate": self.learning_rate,
            "loss": self.loss.name,
            "metrics": [metric.name for metric in self.metrics],
            "epochs": self.epochs,
            "batch_size": self.batch_size
        }

        return "Training Parameters: " + parameters.__repr__() + "\n"

    def save_config(self, file):
        print(self.__repr__(), file=file)


def mlp_multiple_trials(max_runs, train_X, train_y, test_X, test_y, parameters, feature_nums, target_dir):
    
    loss_test = []
    AUPRC_test = []
    models = []

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='AUPRC', patience=20)

    for i in range(max_runs):
        print('randam state', i)
        random.seed(42*i)
        train_X1, train_y1 = shuffle(train_X, train_y, random_state=42*i)

        model = Baseline_MLP(feature_nums)
        # change the number of neurons accordingly
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=parameters.learning_rate),
                  loss=parameters.loss,
                  metrics=parameters.metrics)
        
        history = model.fit(
                train_X1, train_y1,
                epochs=parameters.epochs,
                verbose=2,
                callbacks=[early_stopping_cb]
            )
        
        models.append(model)

        test_X, test_y = shuffle(test_X, test_y, random_state=42*i)
        test_result = model.evaluate(test_X, test_y, verbose=0)
        
        loss_test.append(test_result[0]) 
        AUPRC_test.append(test_result[3]) 
        
        print('test result:', test_result)

    dump_record(target_dir, models, AUPRC_test, loss_test, feature_nums, parameters)


def dump_record(target_dir, models, AUPRC_test, loss_test, feature_nums, parameters):
    os.makedirs(target_dir, exist_ok=True)

    with open(os.path.join(target_dir, "record.txt"), 'w+') as f:
        f.write("Number of neurons: {}\n".format(feature_nums))
        f.write("Test AUPRC: {}\n".format(AUPRC_test))

        parameters.save_config(f)
    
    file = open(os.path.join(target_dir, 'mlp_model_test_scores.save'), 'wb')
    pickle.dump(AUPRC_test, file)
    file.close()

    file = open(os.path.join(target_dir, 'mlp_model_test_loss.save'), 'wb')
    pickle.dump(loss_test, file)
    file.close()

    file = open(os.path.join(target_dir, 'mlp_model_test_model.save'), 'wb')
    pickle.dump(models, file)
    file.close()
