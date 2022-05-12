#! -*- coding: utf-8 -*-
# the capsule parts refer to https://github.com/bojone/Capsule 

import numpy as np
from Capsule_Keras import *
from keras import utils
from keras.models import Model
from keras.layers import *
from keras.metrics import *
from keras import backend as K
from sklearn.model_selection import train_test_split
import sys
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from loss_visualization import visualize_plots
import pickle

# configuration
parser = argparse.ArgumentParser(description='scCapsNet')
# system config
parser.add_argument('--inputdata', type=str, default='dca_2k.npy', help='address for input data')
parser.add_argument('--inputcelltype', type=str, default='dca_2k_celltype.npy', help='address for celltype label')
parser.add_argument('--num_classes', type=int, default=2, help='number of cell type')
parser.add_argument('--randoms', type=int, default=42, help='random number to split dataset')
args = parser.parse_args()


inputdata = args.inputdata
inputcelltype = args.inputcelltype
num_classes = args.num_classes
randoms = args.randoms

data = np.load(inputdata)
labels = np.load(inputcelltype)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size = 0.2, random_state= randoms)
Y_test = y_test
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)

# z_dim = 32

input_size = x_train.shape[1]
x_in = Input(shape=(input_size,))
x = x_in
x1 = Dense(512, activation='relu')(x_in)
# x1 = Dropout(0.2)(x1)
x2 = Dense(128, activation='relu')(x1)
# x2 = Dropout(0.2)(x2)
x3 = Dense(64, activation='relu')(x2)
# # x3 = Dropout(0.2)(x3)
x4 = Dense(32, activation='relu')(x3)

x = Reshape((8, 4))(x4)
num_capsule = 8
capsule = Capsule(num_classes, num_capsule, 3, False)(x) 
output = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)), output_shape=(num_classes,))(capsule)

model = Model(inputs=x_in, outputs=output)

def auroc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

model.compile(loss=lambda y_true, y_pred: y_true*K.relu(0.9-y_pred)**2 + 0.5*(1-y_true)*K.relu(y_pred-0.1)**2 ,
              optimizer='adam',
              metrics=[binary_accuracy, auroc])
model.summary()

history = model.fit(x_train, y_train,
          batch_size=600,
          epochs=150,
          verbose=1,
          validation_data=(x_test, y_test))

model.save_weights('Modelweight.weight')

file = open('./results/dca_2k.save', 'wb')
pickle.dump(model, file)
file.close()

visualize_plots(history, './')


####################################
Y_pred = model.predict(x_test)

coupling_coefficients_value = {}
count = {}
for i in range(len(Y_pred)):
    ind = int(Y_test[i])
    if ind in coupling_coefficients_value.keys():
        coupling_coefficients_value[ind] = coupling_coefficients_value[ind] + Y_pred[i]
        count[ind] = count[ind] + 1
    if ind not in coupling_coefficients_value.keys():
        coupling_coefficients_value[ind] = Y_pred[i]
        count[ind] = 1

total = np.zeros((num_classes,num_capsule))

plt.figure(figsize=(20,np.ceil(num_classes/4)*4))
for i in range(num_classes):
    average = coupling_coefficients_value[i]/count[i]
    Lindex = i + 1
    plt.subplot(np.ceil(num_classes/4),4,Lindex)
    total[i] = average[i]
    df = DataFrame(np.asmatrix(average))
    heatmap = sns.heatmap(df)
plt.savefig("FE_Model_analysis_1_heatmap.png")
plt.show()

###################################################################################################
#overall heatmap
plt.figure()
df = DataFrame(np.asmatrix(total))
heatmap = sns.heatmap(df)

plt.ylabel('Type capsule', fontsize=10)
plt.xlabel('Primary capsule', fontsize=10)
plt.savefig("FE_Model_analysis_1_overall_heatmap.png")
plt.show()
