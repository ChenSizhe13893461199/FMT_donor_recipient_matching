# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 15:33:20 2023

@author: Sizhe Chen
"""
#The following part will introduce the required packages
import os
import string
import torch
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, AveragePooling1D
from keras.layers.pooling import GlobalAveragePooling2D, GlobalAveragePooling1D

#from keras.layers import Input, merge, Flatten
from keras.layers import Input
from keras.layers.reshaping import Flatten
from keras.layers import concatenate, add
from keras.layers.normalization import batch_normalization
from keras.layers import BatchNormalization
from keras.regularizers import l2
import keras.backend as K
from keras.layers import Conv1D, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from utils1 import Phos1, plot_ROC
from keras.optimizers import adam_v2
from utils1 import channel_attenstion
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
import csv
import numpy as np
import keras.utils.np_utils as kutils
from keras.optimizers import adam_v2
from keras.layers import Conv1D, Conv2D, MaxPooling2D
from keras.regularizers import l2
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.pooling import AveragePooling2D, AveragePooling1D
#from keras.layers import Input, merge, Flatten
from keras.layers import Input
from keras.layers.reshaping import Flatten
from keras.layers import concatenate, add
from keras.models import Sequential, Model
import numpy as np
import keras.utils.np_utils as kutils
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
from scipy.special import kl_div


#标签
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


#The core parameter and model file is available in utils1.py file, in which detailed
#model structures as well as mathematical principles are made publicly available.

#The 1st step
#loading the information of FMT donors and recipients, and corresponding labels 
#(training dataset)

donor_training=np.load(file="donor_training.npy")#features of donor
recipient_training=np.load(file="recipient_training.npy")#features of recipient

#The 2nd step
#loading the label of FMT donor-recipient matching (binary format)
#(training dataset)
y_train1=np.load(file="label.npy")#labels for training data


#loading the independent validation dataset
donor_validation=np.load(file="donor_validation.npy")#features of donor
recipient_validation=np.load(file="recipient_validation.npy")#features of recipient
# loading validation labels
y_val_reform=np.load(file="validation_label.npy")

#loading the independent test dataset
donor_testing=np.load(file="donor_testing.npy")#features of donor
recipient_testing=np.load(file="recipient_testing.npy")#features of recipient
#loading testing labels
y_testing_reform=np.load(file="testing_label.npy")




#loading MOZAIC parameters
from sklearn.metrics import roc_auc_score

img_dim1 =donor_training.shape[1:]#matrix shape for features of pre-FMT Recipient

img_dim2 = recipient_training.shape[1:]#matrix shape for features of Donor

img_dim3 = 0#this parameter is not used, please ignore

img_dim4 = 0#this parameter is not used, please ignore

img_dim5 = 0#this parameter is not used, please ignore

img_dim6 = 0#this parameter is not used, please ignore

l=515

#hyperparameters
init_form = 'RandomUniform'
learning_rate = 0.0003
nb_dense_block =5
nb_layers = 5
nb_filter = 32
growth_rate = 32
filter_size_block1 = 5
filter_size_block2 = 5
filter_size_block3 = 5
filter_size_block4 = 0
filter_size_block5 = 0
filter_size_block6 = 0
filter_size_ori = 5
dense_number = 5
dropout_rate = 0.2
dropout_dense = 0.2
weight_decay = 0.000001
nb_batch_size = 2
nb_classes = 2

import tensorflow as tf
from tensorflow.keras import backend as K

model1 = Phos1(nb_classes, nb_layers, img_dim1, img_dim2, init_form, nb_dense_block,growth_rate, 
               filter_size_block1, filter_size_block2, filter_size_block3,filter_size_block4,filter_size_block5,filter_size_block6,
               nb_filter, filter_size_ori,dense_number, dropout_rate, dropout_dense, weight_decay)

opt = adam_v2.Adam(learning_rate=learning_rate,
                   beta_1=0.9, beta_2=0.999, epsilon=1e-08)


from tensorflow.keras.metrics import AUC
model1.compile(loss='binary_crossentropy',optimizer=opt,metrics=[AUC()])
#model1.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])
#if you want to mintor the training of MOZAIC by using accuracy indicator, please use
#the aforementioned codes instead




#the training process
#defining epoch numbers
epochs=30
x11=42#validation dataset size
x12=103#test dataset size
y_probs=np.zeros(shape=(x11,1))
y_probst=np.zeros(shape=(x12,1))
y_t=np.zeros(shape=(x12,1))
for i in range(0,epochs): 
  nb_epoch = 1#every little step for collecting information in MOZAIC training
  history = model1.fit([donor_training,recipient_training], y_train1, batch_size=nb_batch_size, validation_split=0,epochs=nb_epoch, shuffle=True, verbose=1)
  #validation step is set as 0, as we have additionally set independent validation dataset
  y_probs[0:x11,0] = (model1.predict([donor_validation,recipient_validation])[:,0]).flatten()
  fold_auc = roc_auc_score(y_val_reform, y_probs)


  #monitoring the training process for avoiding over-fitting
  print(f"Validation Dataset increased microbiome AUC: {fold_auc:.4f}")

#after you have finished model training, you can test the efficacy of MOZAIC on the
#independent testing dataset
#model evaluations on the independent testing dataset
y_probst[0:x12,0] = (model1.predict([donor_testing,recipient_testing])[:,0]).flatten()
y_t[0:x12,0]=y_testing_reform
fold_auct = roc_auc_score(y_t, y_probst)

print(f"Test Dataset increased microbiome AUC: {fold_auct:.4f}")

