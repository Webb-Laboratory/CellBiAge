import tensorflow as tf


class Baseline_MLP(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense((64), activation='relu'),
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense((32), activation='relu'),
            tf.keras.layers.Dense((16), activation='relu'),
            tf.keras.layers.Dense((1), activation='sigmoid')
        ])

    def call(self, inputs):
        prob = self.mlp(inputs)
        return prob
