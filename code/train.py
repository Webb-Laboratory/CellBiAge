import pickle
import tensorflow as tf
import os
from loss_visualization import visualize_plots


class train_params:
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


def train(target_dir, model, parameters, train_X, train_y, test_X, test_y, dataset_name):
    os.makedirs(target_dir, exist_ok=True)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=parameters.learning_rate),
                  loss=parameters.loss,
                  metrics=parameters.metrics)

    history = model.fit(
        train_X, train_y,
        validation_data=(test_X, test_y),
        epochs=parameters.epochs,
        batch_size=parameters.batch_size,
        verbose=2
    )

    testing_result = model.evaluate(test_X, test_y, verbose=0)
    dump_record(target_dir, model, parameters, testing_result, dataset_name)
    visualize_plots(history, target_dir)


def dump_record(target_dir, model, parameters, testing_result, dataset_name):
    with open(os.path.join(target_dir, "record.txt"), 'w+') as f:
        f.write("Testing Score: {}\n".format(testing_result))
        f.write("Dataset: {}\n".format(dataset_name))
        parameters.save_config(f)

    model.save(os.path.join(target_dir, "model"))
