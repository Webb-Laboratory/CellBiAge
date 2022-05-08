import tensorflow as tf


class Baseline_MLP(tf.keras.Model):

    def __init__(self, feature_nums=None, dropout_rate=0, initializer=tf.keras.initializers.HeNormal(), **kwargs):
        super().__init__(**kwargs)
        if feature_nums is None:
            feature_nums = [64, 32, 16]

        self.initializer = initializer

        self.mlp = tf.keras.Sequential()
        for i in feature_nums:
            self.mlp.add(tf.keras.layers.Dense(i, activation='relu', kernel_initializer=self.initializer))
            if dropout_rate > 0:
                self.mlp.add(tf.keras.layers.Dropout(dropout_rate))
       
        #self.mlp = tf.keras.Sequential([tf.keras.layers.Dense(i, activation='relu', kernel_initializer=self.initializer) for i in feature_nums] )                              
                                
        self.pred_head = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=self.initializer)


    def call(self, inputs):
        x = self.mlp(inputs)
        prob = self.pred_head(x)
        return prob
