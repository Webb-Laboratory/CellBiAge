import os

from matplotlib import pyplot as plt

def visualize_plots(history, store_dir=None):
    """
    Uses Matplotlib to visualize the losses and metrics of our model.
    :param history: dictionary of losses and metrics for train and validation data return from model.fit()
    :return: doesn't return anything, three plots should pop-up
    """

    # summarize history for accuracy
    plt.plot(history.history['binary_accuracy'])
    plt.plot(history.history['val_binary_accuracy'])
    plt.title('model binary_accuracy')
    plt.ylabel('binary_accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    if store_dir:
        plt.savefig(os.path.join(store_dir, "accuracy_plot.png"))
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    if store_dir:
        plt.savefig(os.path.join(store_dir, "loss_plot.png"))
    plt.show()

    # summarize history for AUC
    plt.plot(history.history['auc'])
    plt.plot(history.history['val_auc'])
    plt.title('model auc')
    plt.ylabel('auc')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    if store_dir:
        plt.savefig(os.path.join(store_dir, "auc_plot.png"))
    plt.show()