import tensorflow as tf

class Dense_AE(tf.keras.Model):
    def __init__(self, latent_dim = 64, **kwargs):
        super().__init__(**kwargs)#super(Autoencoder, self).__init__()
        
        self.latent_dim = latent_dim   
        self.encoder = tf.keras.Sequential([
        tf.keras.layers.Dense(latent_dim, activation='relu'),
        tf.keras.layers.Dense(latent_dim,  activation='relu')
        ])

        self.decoder = tf.keras.Sequential([
        tf.keras.layers.Dense(latent_dim,  activation='relu'),
        tf.keras.layers.Dense(latent_dim,  activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        result = self.encoder(x)
        return result


