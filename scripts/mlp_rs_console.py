from src.keras_tuner_cv import *
from src.data_processing import *
from src.mlp import *

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
@click.option('--max_runs', default=10, 
    help='how many random states?', show_default=True)
@click.option('--cell_type', default='All', 
    prompt='to choose among: \n All, Non-neuronal, Neuron, \n Oligodendrocyte, Astrocyte, Microglia, \n VLMC, OPC, Tanycyte, Ependymocyte, \n Endothelial Cell, Pericyte, Macrophage',
       help='which cell type to use?', show_default=True)
@click.option('--learning_rate', default=3.1178891827106297e-05, help='the learning rate of the model', show_default=True)
@click.option('--feature_nums', default=[256], multiple=True, help='the number of neruons per layer, and the number of layers', show_default=True)

def main(cuda_device, input_test, input_train, cell_type, output_dir, max_runs, feature_nums, learning_rate, binarization=True):

    parameters = Train_Params(
    learning_rate=learning_rate,
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC(), 
    tf.keras.metrics.AUC(num_thresholds=10000, name='AUPRC', curve='PR')],
    epochs=100,
    batch_size=32)

    print(cuda_device, input_test, input_train, cell_type, output_dir, binarization)

    os.environ["CUDA_VISIBLE_DEVICES"]=cuda_device
    
    train_X, train_y, test_X, test_y, custom_cv = data_prep(input_test, input_train, 
                                                            cell_type, binarization=True)
    
    directory=os.path.join(output_dir, 'mlp_rs_'+ str(cell_type) + '_'+ datetime.datetime.now().strftime('%m-%d_%H-%M-%S'))
    print('the mlp results are stored in ', directory)
    
    mlp_multiple_trials(max_runs, train_X, train_y, test_X, test_y, parameters, feature_nums, directory)    
        
if __name__ == '__main__':
    main()


