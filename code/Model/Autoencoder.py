import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

latent_dim = None 
class Autoencoder(Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    if latent_dim is None:
        latent_dim = 64
    self.latent_dim = latent_dim   
    self.encoder = tf.keras.Sequential([
      layers.Dense(latent_dim, activation='relu'),
      layers.Dense(latent_dim,  activation='relu')
    ])

    self.decoder = tf.keras.Sequential([
      layers.Dense(latent_dim,  activation='relu'),
      layers.Dense(latent_dim,  activation='relu'),
      layers.Dense(1, activation='sigmoid')
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded