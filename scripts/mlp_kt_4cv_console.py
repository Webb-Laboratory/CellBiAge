import src
from src import *
import keras_tuner as kt
import pandas as pd
import numpy as np
import os
import click
import datetime

@click.command()
@click.option('--cuda_device', default='0', help='which GPU to use', show_default=True)
@click.option('--input_train', default='../data/train_final_group_info.csv', 
    help='training set dir', show_default=True)
@click.option('--input_test', default='../data/test_final_group_info.csv', 
    help='test set dir', show_default=True)
@click.option('--output_dir', default='../results/', 
    help='output dir', show_default=True)
@click.option('--binarization', default=True, 
    help='binarization?', show_default=True)
@click.option('--max_trials', default=50, 
    help='how many trials?', show_default=True)
@click.option('--cell_type', default='Neuron', 
    prompt='to choose among: \n All, Non-neuronal, Neuron, \n Oligodendrocyte, Astrocyte, Microglia, \n VLMC, OPC, Tanycyte, Ependymocyte, \n Endothelial Cell, Pericyte, Macrophage',
       help='which cell type to use?', show_default=True)

def main(cuda_device, input_test, input_train, cell_type, output_dir, max_trials, binarization=True):
    
    print(cuda_device, input_test, input_train, cell_type, output_dir, binarization)
    
    os.environ["CUDA_VISIBLE_DEVICES"]=cuda_device
    
    train_X, train_y, test_X, test_y, custom_cv = data_prep(input_test, input_train, 
                                                            cell_type, binarization=True)
    
    directory=os.path.join(output_dir, 'cv_'+ cell_type + '_'+ datetime.datetime.now().strftime('%m-%d_%H-%M-%S'))
    
    print('the cv results are stored in ', directory)
    tuner = CVTuner(
        hypermodel = build_model,
        oracle = kt.oracles.RandomSearch(kt.Objective("val_AUPCR", direction="max"),
                                         max_trials=max_trials, seed=84),
        overwrite=True,
        directory=directory)

    tuner.search(train_X, train_y, custom_cv)

    best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
    best_trial.summary()

    hps = best_trial.hyperparameters
    model = build_model(hps)

    history = model.fit(train_X, train_y,
                        epochs=50,
                        batch_size=32,
                        verbose=2)

    testing_result = model.evaluate(test_X, test_y, verbose=0)
    print(testing_result)
    
        
if __name__ == '__main__':
    main()