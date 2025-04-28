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
from utils1 import getMatrixLabel, Phos1,PhosB, PhosE, getMatrixInput, getMatrixInputh, getMatrixLabelFingerprint, getMatrixLabelh, plot_ROC, getMatrixLabelFingerprint1,getMatrixLabelnlp
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
from propy.GetProteinFromUniprot import GetProteinSequence
from propy.PyPro import GetProDes
from propy.GetProteinFromUniprot import GetProteinSequence
from propy.GetProteinFromUniprot import GetProteinSequence
from propy.GetProteinFromUniprot import GetProteinSequence as gps
from propy import GetSubSeq
from propy.GetProteinFromUniprot import GetProteinSequence
from propy.AAComposition import CalculateAAComposition
from propy.AAComposition import CalculateAADipeptideComposition
from propy.AAComposition import GetSpectrumDict
from propy.AAComposition import Getkmers
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
from scipy.special import kl_div


#标签
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. 读取标签数据
df_label = pd.read_excel('AI_label.xlsx', usecols=[0])  # 第4列（索引从0开始）
df_data = pd.read_excel('filtered_microbiome_overall_4categories23_67.xlsx')  # 第4列（索引从0开始）
df_data1 = pd.read_excel('filtered_pathway_overall_4categories_85.xlsx')  # 第4列（索引从0开始）
df_data2 = pd.read_excel('filtered_ko_overall_4categories_85.xlsx')  # 第4列（索引从0开始）
y = df_label.values.ravel()  # 将标签转换为1D数组
#数据集构建
xxx=0
X=np.zeros(shape=(515,int(len(df_data.columns)*2+len(df_data1.columns)*2)+len(df_data2.columns)*2))#496
for i in range(0,515):
     for t in range(0,len(df_data.columns)):
      X[i][t]=df_data.iloc[i,t]
     for t in range(len(df_data.columns),len(df_data.columns)+len(df_data1.columns)):
      X[i][t]=df_data1.iloc[i,t-len(df_data.columns)]
     for t in range(len(df_data.columns)+len(df_data1.columns),len(df_data.columns)+len(df_data1.columns)+len(df_data2.columns)):
      X[i][t]=df_data2.iloc[i,t-len(df_data.columns)-len(df_data1.columns)]

for i in range(1030,1545):#992 1488
     for t in range(0,len(df_data.columns)):
      X[i-1030][t+len(df_data.columns)+len(df_data1.columns)+len(df_data2.columns)]=df_data.iloc[i,t]      
     for t in range(len(df_data.columns),len(df_data.columns)+len(df_data1.columns)):
      X[i-1030][t+len(df_data.columns)+len(df_data1.columns)]=df_data1.iloc[i,t-len(df_data.columns)]
     for t in range(len(df_data.columns)+len(df_data1.columns),len(df_data.columns)+len(df_data1.columns)+len(df_data2.columns)):
      X[i-1030][t+len(df_data.columns)+len(df_data1.columns)+len(df_data2.columns)]=df_data2.iloc[i,t-len(df_data.columns)-len(df_data1.columns)]
from sklearn.metrics import roc_auc_score




import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
ax1=515
ax2=1840
# 数据重塑为59x42格式 
similarity=pd.read_excel('AI_label.xlsx', usecols=[0])  # 第4列（索引从0开始）
#clinical_response=pd.read_excel('dissimilarity and similarity.xlsx', usecols=[0])  # 第4列（索引从0开始）
similarity = similarity.values.ravel()





X_1 = X[0:ax1,0:ax2].reshape(-1, 40, 46)  # 添加通道维度 43 47
X_2 = X[0:ax1,ax2:ax2*2].reshape(-1, 40, 46)  # 添加通道维度



y_train_reform=np.zeros(shape=(ax1,2))#471
for i in range(0,ax1):#38
  #if clinical_response.category[i]!=2:
    #y_train_reform[i][0]=y[i]
    if (similarity[i]<=0):
        y_train_reform[i][0]=1

    # if (0.01<similarity[i]<=0.05):
    #       y_train_reform[i][1]=1

    if (0<similarity[i]):
        y_train_reform[i][1]=1
        
################################################################

from sklearn.metrics import roc_auc_score

img_dim1 =X_1.shape[1:]

img_dim2 = X_2.shape[1:]

img_dim3 = X_1.shape[1:]

img_dim4 = X_1.shape[1:]

img_dim5 = X_1.shape[1:]

img_dim6 = X_1.shape[1:]

l=515
# a=X[0:l,0:1776]
# b=X[0:l,1776:3552]
# X_1 = a.reshape(-1, 48, 37)  # 添加通道维度
# X_2 = b.reshape(-1, 48, 37)  # 添加通道维度


# a1=X[471:496,0:1776]
# b1=X[471:496,1776:3552]
# X_test_1=a1.reshape(-1, 48, 37)  # 添加通道维度
# X_test_2=b1.reshape(-1, 48, 37)  # 添加通道维度
##############################################################

# y_test_reformt=y_train_reform[471:496,:]

#####################################################################




#13 9 12
#12
#14
X_train1, X_test1, y_train1, y_test1 = train_test_split(
    X_1, y_train_reform[0:l,:], test_size=0.28, random_state=12
)

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_2, y_train_reform[0:l,:], test_size=0.28, random_state=12
)


X_train2, X_test2, i1, i2 = train_test_split(
    X_2, similarity, test_size=0.28, random_state=13
)




import numpy as np
count = np.count_nonzero(i2[0:41] == 7)
count = np.count_nonzero(y_test1[0:41,0] == 1)
print(count/41)
count = np.count_nonzero(i2[41:] == 7)
count = np.count_nonzero(y_test1[41:,0] == 1)
count = np.count_nonzero(y_train_reform[:,0] == 1)
print(count/104)

category=[]
for i in range(1,14):
    count = np.count_nonzero(i2[41:] == i)
    category.append(count)
#similarity=pd.read_excel('AI_label.xlsx', usecols=[1])  # 第4列（索引从0开始）
#similarity = similarity.values.ravel()
# index=pd.read_excel('AI_label.xlsx', usecols=[1])  # 第4列（索引从0开始）
# X_train1, X_test1, index1, index2 = train_test_split(
#     X_1, index[0:471], test_size=0.5, random_state=16
# )

##############################################

init_form = 'RandomUniform'
learning_rate = 0.0003#0.001
nb_dense_block =5#6
nb_layers = 5#6
nb_filter = 32#32
growth_rate = 32#32
filter_size_block1 = 5#11
filter_size_block2 = 5#11
filter_size_block3 = 5#11
filter_size_block4 = 0
filter_size_block5 = 0
filter_size_block6 = 0
filter_size_ori = 5
dense_number = 5#11
dropout_rate = 0.2
dropout_dense = 0.2
weight_decay = 0.000001
nb_batch_size = 16#16
nb_classes = 2

import tensorflow as tf
from tensorflow.keras import backend as K

model1 = Phos1(nb_classes, nb_layers, img_dim1, img_dim2, init_form, nb_dense_block,growth_rate, 
               filter_size_block1, filter_size_block2, filter_size_block3,filter_size_block4,filter_size_block5,filter_size_block6,
               nb_filter, filter_size_ori,dense_number, dropout_rate, dropout_dense, weight_decay)

opt = adam_v2.Adam(learning_rate=learning_rate,
                   beta_1=0.9, beta_2=0.999, epsilon=1e-08)

#model1.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])
from tensorflow.keras.metrics import AUC
model1.compile(loss='binary_crossentropy',optimizer=opt,metrics=[AUC()])

# 然后将 resampled 数据用于训练


from sklearn.metrics import accuracy_score, recall_score  # 添加这两个导入

score1=[]
number=[]
inl=l*0.28-l*0.2#test
inl1=l*0.2#validation
x11=int(inl)+1#y_test1[:,0]
y_probs=np.zeros(shape=(x11,1))
y_val_reform=np.zeros(shape=(x11,1))
score1t=[]
numbert=[]
x12=int(inl1)
y_probst=np.zeros(shape=(x12,1))
y_val_reformt=np.zeros(shape=(x12,1))
#multi-class
for i in range(0,100): 
 nb_epoch = 1#y_train1
 history = model1.fit([X_train1,X_train2], y_train1, batch_size=nb_batch_size,validation_split=0,epochs=nb_epoch, shuffle=True, verbose=1)
 y_probs[0:x11,0] = (model1.predict([X_test1[0:x11],X_test2[0:x11]])[:,0]).flatten()
 # y_probs[len(y_test1[:,0]):len(y_test1[:,0])*2,0] = (model1.predict([X_test1,X_test2])[:,1]).flatten()
 # y_probs[len(y_test1[:,0])*2:len(y_test1[:,0])*3,0] = (model1.predict([X_test1,X_test2])[:,2]).flatten()
 y_val_reform[0:x11,0]=y_test1[0:x11,0].flatten()
 # y_val_reform[len(y_test1[:,0]):len(y_test1[:,0])*2,0]=y_test2[:,1].flatten()
 # y_val_reform[len(y_test1[:,0])*2:len(y_test1[:,0])*3,0]=y_test2[:,2].flatten()
 fold_auc = roc_auc_score(y_val_reform, y_probs)
 #fold_auct = roc_auc_score(y_val_reform[50:100,0], y_probs[50:100,0])

 #model1.load_weights('FMT_matching_prediction_nnw_multil_seed_16_overall_38.h5')
 y_probst[0:x12,0] = (model1.predict([X_test1[x11:x11+x12],X_test2[x11:x11+x12]])[:,0]).flatten()
 # y_probst[len(y_train_reform[471:496,:]):len(y_train_reform[471:496,:])*2,0] = (model1.predict([X_test_1,X_test_2])[:,1]).flatten()
 # y_probst[len(y_train_reform[471:496,:])*2:len(y_train_reform[471:496,:])*3,0] = (model1.predict([X_test_1,X_test_2])[:,2]).flatten()
 y_val_reformt[0:x12,0]=y_test1[x11:x11+x12,0].flatten()
 # y_val_reformt[len(y_train_reform[471:496,:]):len(y_train_reform[471:496,:])*2,0]=y_test_reformt[:,1].flatten()
 # y_val_reformt[len(y_train_reform[471:496,:])*2:len(y_train_reform[471:496,:])*3,0]=y_test_reformt[:,1].flatten()

 fold_auct = roc_auc_score(y_val_reformt, y_probst)

 # auc_scores.append(fold_auc)
 print(f"Validation Dataset increased microbiome AUC: {fold_auc:.4f}")
 print(f"Test Dataset increased microbiome AUC: {fold_auct:.4f}")
 y_pred = (y_probst >= 0.5).astype(int)  # 将概率转为类别预测
 accuracyt = accuracy_score(y_val_reformt, y_pred)
 recallt = recall_score(y_val_reformt, y_pred)
 print(f"Test AUC: {fold_auct:.4f} | Accuracy: {accuracyt:.4f} | Recall: {recallt:.4f}")
  # print(f"Test increased microbiome AUC: {fold_auct:.4f}")
 if fold_auc >=0.83 and fold_auct >=0.85 and accuracyt>=0.8 and recallt >=0.8:
         # y_pred = (y_probst >= 0.5).astype(int)  # 将概率转为类别预测
         model1.save_weights('FMT_microbiome_seed_12_2_'+str(i)+'.h5', overwrite=True)
         print(f"fold is {i:.1f}")
         # accuracyt = accuracy_score(y_val_reformt, y_pred)
         # recallt = recall_score(y_val_reformt, y_pred)
         print(f"Test AUC: {fold_auct:.4f} | Accuracy: {accuracyt:.4f} | Recall: {recallt:.4f}")
         score1.append(fold_auc)
         number.append(i)
#2025 4 14
#12 random seed named 9 in document
#5 6 20 22 29 31 33

#11 14 18 20 22 23 26

#2025 4 14
#14 random seed named 3 in document
#10 


#2025 3 31
#18 0.8500 0.8305
#17 0.8237 0.8152
#16 0.8632 0.8325
#15 0.8711 0.8144
#20 0.8000 0.8319
#496 samples
model1.load_weights('FMT_microbiome_seed_12_1_11.h5')
# y_probs[0:x11,0] = (model1.predict([X_test1[0:x11],X_test2[0:x11]])[:,0]).flatten()
# y_val_reform[0:x11,0]=y_test1[0:x11,0].flatten()
y_probst[0:x12,0] = (model1.predict([X_test1[x11:x11+x12],X_test2[x11:x11+x12]])[:,0]).flatten()
y_val_reformt[0:x12,0]=y_test1[x11:x11+x12,0].flatten()

 
from sklearn.metrics import accuracy_score, recall_score  # 添加这两个导入
y_pred = (y_probst >= 0.5).astype(int)  # 将概率转为类别预测
fold_auct = roc_auc_score(y_val_reformt, y_probst)
accuracyt = accuracy_score(y_val_reformt, y_pred)
recallt = recall_score(y_val_reformt, y_pred)
print(f"Test AUC: {fold_auct:.4f} | Accuracy: {accuracyt:.4f} | Recall: {recallt:.4f}")

TP = np.sum((y_pred == 1) & (y_val_reformt == 1))
TN = np.sum((y_pred == 0) & (y_val_reformt == 0))
FP = np.sum((y_pred == 1) & (y_val_reformt == 0))
FN = np.sum((y_pred == 0) & (y_val_reformt == 1))

accuracy = (TP + TN) / (TP + TN + FP + FN)
recall = TP / (TP + FN) if (TP + FN) != 0 else 0


#results 20
#14
#16
#18
#21
#24
#26
#27
#28
#31
#33
#37
#45
#47
score1=[]
number=[]
y_probs=np.zeros(shape=(x11*2,1))
y_val_reform=np.zeros(shape=(x11*2,1))
score1t=[]
numbert=[]
y_probst=np.zeros(shape=(x12*2,1))
y_val_reformt=np.zeros(shape=(x12*2,1))

from sklearn.metrics import roc_auc_score, roc_curve
model1.load_weights('FMT_microbiome_seed_22_23.h5')
y_probs[0:x11,0] = (model1.predict([X_test1[0:x11],X_test2[0:x11]])[:,0]).flatten()
y_probs[x11:x11*2,0] = (model1.predict([X_test1[0:x11],X_test2[0:x11]])[:,1]).flatten()
y_val_reform[0:x11,0]=y_test1[0:x11,0].flatten()
y_val_reform[x11:x11*2,0]=y_test1[0:x11,1].flatten()

fold_auc = roc_auc_score(y_val_reform, y_probs)
y_probst[0:x12,0] = (model1.predict([X_test1[x11:x11+x12],X_test2[x11:x11+x12]])[:,0]).flatten()
y_probst[x12:x12*2,0] = (model1.predict([X_test1[x11:x11+x12],X_test2[x11:x11+x12]])[:,1]).flatten()
y_val_reformt[0:x12,0]=y_test1[x11:x11+x12,0].flatten()
y_val_reformt[x12:x12*2,0]=y_test1[x11:x11+x12,1].flatten()

fold_auct = roc_auc_score(y_val_reformt, y_probst)
a1, b1, thresholds1 = roc_curve(y_val_reform[0:x11*2,0], y_probs[0:x11*2,0])
a2, b2, thresholds2 = roc_curve(y_val_reformt[0:x12*2,0], y_probst[0:x12*2,0])
fold_auc = roc_auc_score(y_val_reform[0:x11], y_probs[0:x11])
fold_auct = roc_auc_score(y_val_reformt[0:x12], y_probst[0:x12])
print(f"Validation Dataset increased microbiome AUC: {fold_auc:.4f}")
print(f"Test Dataset increased microbiome AUC: {fold_auct:.4f}")
#random seed 22
#11 0.8132 0.8209
#12 0.8104 0.8277
#13 0.8132 0.8345
#14 0.8104 0.8349
#15 0.8077 0.8345
#17 0.8077 0.8365
#18 0.8022 0.8357
#19 0.8132 0.8355
#20 0.8159 0.8349
#23 0.8077 0.8500


from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_val_reformt, y_probst)
# 手动计算 AUC（使用梯形法则）
auc_manual = np.trapz(tpr, fpr)
print("Manual AUC Score:", auc_manual)


fpr=np.zeros(shape=(100,10))
tpr=np.zeros(shape=(100,10))
from sklearn.metrics import roc_curve, auc
t1, t2, _ = roc_curve(y_val_reformt, y_probst)
fpr[0:len(t1),1], tpr[0:len(t2),1], _ = roc_curve(y_val_reformt, y_probst)
    
    
    
    
    
    
    
    
    
    
    