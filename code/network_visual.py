import tensorflow as tf
from matplotlib import pyplot as plt
from dataset import *


def main():
    train_X, test_X, train_y, test_y, _ = without_cate_binarized_dataset()

    model = tf.keras.models.load_model('../results/MLP_layer_100_50_10_with_binarized_lr_001_without_cate/model')
    weight, bias = model.mlp.layers[0].get_weights()

    plt.plot(weight)
    plt.show()


if __name__ == '__main__':
    main()
