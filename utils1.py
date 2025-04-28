#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 11:26:27 2022

@author: Sizhe Chen
"""

import csv
import numpy as np
import keras.utils.np_utils as kutils
# from keras.optimizers import Adam, SGD
from keras.optimizers import adam_v2
from keras.layers import Conv1D,Conv2D, MaxPooling2D,MaxPooling1D,GlobalMaxPooling1D
from keras.regularizers import l2
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.pooling import AveragePooling2D, AveragePooling1D,GlobalAveragePooling1D
#from keras.layers import Input, merge, Flatten
from keras.layers import Input
from keras.layers.reshaping import Flatten
from keras.layers import concatenate, add
#from keras.layers import Input, merge, Flatten
from keras.models import Sequential, Model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import math
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

######################################
#自注意力机制
import torch
import numpy as np
import math
import torch.nn.functional as F




import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras import Model


###################################################
import tensorflow as tf
from tensorflow import keras

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def encoder(input_tensor):
    d_model = input_tensor.shape[-1]
    num_heads = 5
    dff = 128
    rate = 0.1

    # Multi-head self-attention
    attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model, dropout=rate)(input_tensor, input_tensor)
    attention = layers.LayerNormalization(epsilon=1e-6)(attention + input_tensor)

    # Feedforward neural network
    ffn = keras.Sequential([
        layers.Dense(dff, activation="relu"),
        layers.Dense(d_model)
    ])
    ffn_output = ffn(attention)
    encoder_output = layers.LayerNormalization(epsilon=1e-6)(ffn_output + attention)

    return encoder_output

#Transformer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def encoder1(input_tensor):
    d_model = input_tensor.shape[-1]#every input is described as a d_model size vector
    num_heads = 10#9
    dff = 128
    rate = 0.1
    
    # Multi-head self-attention
    q = layers.Dense(d_model)(input_tensor)
    k = layers.Dense(d_model)(input_tensor)
    v = layers.Dense(d_model)(input_tensor)
    
    attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model, dropout=rate)(q, k, v)
    attention = layers.LayerNormalization(epsilon=1e-6)(attention + input_tensor)
    
    # Feedforward neural network
    ffn = keras.Sequential([
        layers.Dense(dff, activation="relu"),
        layers.Dense(d_model)
    ])
    
    ffn_output = ffn(attention)
    encoder_output = layers.LayerNormalization(epsilon=1e-6)(ffn_output + attention)
    
    return encoder_output

###################################################
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dense, Activation, Multiply, Input
from tensorflow.keras.models import Model


import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Activation, Reshape, Multiply, Lambda, Concatenate

def synthetic_attention1(inputs):
    # 计算权重
   outputs=[]
   num_heads=9
   for _ in range(num_heads):
    weights = tf.keras.layers.Dense(1, activation='tanh')(inputs)
    weights = tf.keras.layers.Flatten()(weights)
    weights = tf.keras.layers.Activation('softmax')(weights)
    weights = tf.keras.layers.Reshape((-1, 1))(weights)
    
    weighted_inputs = Multiply()([inputs, weights])
    output = Lambda(lambda x: tf.keras.backend.sum(x, axis=1))(weighted_inputs)
    outputs.append(output)
    
    # 加权求和
    weighted_inputs = tf.keras.layers.Multiply()([inputs, weights])
    output = tf.keras.layers.Lambda(lambda x: tf.keras.backend.sum(x, axis=1))(weighted_inputs)
    
    return output

#######################################
import torch
import torch.nn as nn
import torchvision


import torch
import torch.nn as nn

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Reshape, Lambda, Permute, Multiply, Add

import tensorflow as tf
from tensorflow.keras import layers

def non_local_attention(input_tensor):
    _, width, filters = input_tensor.shape.as_list()

    theta = layers.Conv1D(filters=filters // 8, kernel_size=1, strides=1)(input_tensor)
    theta = tf.reshape(theta, shape=(-1, width, filters // 8))

    phi = layers.Conv1D(filters=filters // 8, kernel_size=1, strides=1)(input_tensor)
    phi = tf.reshape(phi, shape=(-1, width, filters // 8))

    g = layers.Conv1D(filters=filters // 2, kernel_size=1, strides=1)(input_tensor)
    g = tf.reshape(g, shape=(-1, width, filters // 2))

    theta_phi = tf.matmul(theta, phi, transpose_b=True)
    theta_phi = tf.nn.softmax(theta_phi)

    attention = tf.matmul(theta_phi, g)
    attention = tf.reshape(attention, shape=(-1, width, filters // 2))

    output = layers.Conv1D(filters=filters, kernel_size=1, strides=1)(attention)
    output = tf.keras.layers.add([output, input_tensor])

    return output



from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Multiply, Add, Reshape

def attention_mechanism(inputs):
    # 使用全局平均池化和全局最大池化来计算注意力权重
    global_avg_pool = GlobalAveragePooling2D()(inputs)
    global_max_pool = GlobalMaxPooling2D()(inputs)
    
    # 通过全连接层计算注意力权重
    avg_fc = Dense(inputs.shape[-1], activation='relu')(global_avg_pool)
    max_fc = Dense(inputs.shape[-1], activation='relu')(global_max_pool)
    
    # 将两个全连接层的输出相加
    fc = Add()([avg_fc, max_fc])
    
    # 使用sigmoid激活函数计算注意力权重
    attention_weights = Dense(inputs.shape[-1], activation='sigmoid')(fc)
    
    # 将注意力权重重塑为和输入同样的形状，然后用来加权输入
    attention_weights = Reshape((1, 1, inputs.shape[-1]))(attention_weights)
    weighted_inputs = Multiply()([inputs, attention_weights])
    
    return weighted_inputs

from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Conv2D, Multiply, Add, Activation, Reshape

def block3(inputs):

    def channel_attention(input_feature, ratio=8):
        channel_axis = 1 if tf.keras.backend.image_data_format() == "channels_first" else -1
        channel = input_feature.shape[channel_axis]
        
        shared_layer_one = Dense(channel//ratio, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
        shared_layer_two = Dense(channel, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
        
        avg_pool = GlobalAveragePooling1D()(input_feature)    
        avg_pool = Reshape((1,1,channel))(avg_pool)
        assert avg_pool.shape[1:] == (1,1,channel)
        avg_pool = shared_layer_one(avg_pool)
        assert avg_pool.shape[1:] == (1,1,channel//ratio)
        avg_pool = shared_layer_two(avg_pool)
        assert avg_pool.shape[1:] == (1,1,channel)
        
        max_pool = GlobalMaxPooling1D()(input_feature)
        max_pool = Reshape((1,1,channel))(max_pool)
        assert max_pool.shape[1:] == (1,1,channel)
        max_pool = shared_layer_one(max_pool)
        assert max_pool.shape[1:] == (1,1,channel//ratio)
        max_pool = shared_layer_two(max_pool)
        assert max_pool.shape[1:] == (1,1,channel)
        
        cbam_feature = Add()([avg_pool,max_pool])
        cbam_feature = Activation('sigmoid')(cbam_feature)
        p1=channel/ratio * channel + channel * channel
        return Multiply()([input_feature, cbam_feature]),p1

    def spatial_attention(input_feature):
        average_color = tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=3, keepdims=True))(input_feature)
        max_color = tf.keras.layers.Lambda(lambda x: tf.keras.backend.max(x, axis=3, keepdims=True))(input_feature)
        concat = Concatenate(axis=3)([average_color, max_color])
        spatial_attention_feature = Conv2D(filters = 1, kernel_size=7, padding='same', activation='sigmoid')(concat)   
        
        return Multiply()([input_feature, spatial_attention_feature])
    
    x = inputs
    q16=x.shape
    x,p1 = channel_attention(x)
    #q16=x.shape
    x = spatial_attention(x)
    tt=p1+49
    return x,tt,q16

#################################################

# 将原始训练数据转成输入数据的函数
def getMatrixLabel(positive_position_file_name,sites, window_size=51, empty_aa = '*'):
    # input format   label, proteinName, postion, shortsequence
    # label 0/1
    prot = []  #
    pos = []  #
    rawseq = [] #
    all_label = [] #

    short_seqs = []
    half_len = (window_size - 1) / 2

    with open(positive_position_file_name, 'r') as rf:
        reader = csv.reader(rf)
        for row in reader:

                position = int(row[2])
                sseq = row[3]
                rawseq.append(row[3])
                center = sseq[position - 1]
                all_label.append(int(row[0]))
                prot.append(row[1])
                pos.append(row[2])
                # coding = one_hot_concat(shortseq)
                # all_codings.append(coding)
        
        # Keras的utilities，用于“Converts a class vector (integers) to binary class matrix.”
        # “A binary matrix representation of the input”
        ONE_HOT_SIZE = 21
        # _aminos = 'ACDEFGHIKLMNPQRSTVWY*'
        letterDict = {}
        letterDict["A"] = 0
        letterDict["C"] = 1
        letterDict["D"] = 2
        letterDict["E"] = 3
        letterDict["F"] = 4
        letterDict["G"] = 5
        letterDict["H"] = 6
        letterDict["I"] = 7
        letterDict["K"] = 8
        letterDict["L"] = 9
        letterDict["M"] = 10
        letterDict["N"] = 11
        letterDict["P"] = 12
        letterDict["Q"] = 13
        letterDict["R"] = 14
        letterDict["S"] = 15
        letterDict["T"] = 16
        letterDict["V"] = 17
        letterDict["W"] = 18
        letterDict["Y"] = 19
        letterDict["*"] = 20

        #
        Matr = np.zeros((len(rawseq), window_size, ONE_HOT_SIZE))
        samplenumber = 0
        for seq in rawseq:
            AANo = 0
            for AA in seq:
                index = letterDict[AA]
                Matr[samplenumber][AANo][index] = 1
                AANo = AANo+1
            samplenumber = samplenumber + 1

    return Matr

import gensim
from gensim.models import Word2Vec
import numpy as np
# 
def getMatrixLabelh(positive_position_file_name, window_size=51, empty_aa = '*'):
    # input format   label, proteinName, postion, shortsequence
    # label存储0/1值
    #positive_position_file_name='trainingAMP.csv'
    #window_size=50
    prot = []  #
    pos = []  #
    rawseq = [] #
    all_label = [] #
    all_labela = []
    length=[]
    short_seqs = []
    #half_len = (window_size - 1) / 2

    with open(positive_position_file_name, 'r') as rf:
        reader = csv.reader(rf)
        for row in reader:

                #position = int(row[2])
                a=window_size-len(row[1])
                sseq = row[1]#+a*' '
                rawseq.append(sseq)
                b=len(row[1])
                length.append(b)
                #center = sseq[position - 1]
            # 
                all_label.append(row[0])
                #all_labela.append(int(row[2])) 
                #prot.append(row[1])
                #pos.append(row[2])

        
        # Keras的utilities，用于“Converts a class vector (integers) to binary class matrix.”
        # “A binary matrix representation of the input”
        targetY = kutils.to_categorical(all_label)
        #targetYa = kutils.to_categorical(all_labela)

        ONE_HOT_SIZE = 20
        # _aminos = 'ACDEFGHIKLMNPQRSTVWY*'
        letterDict = {}
        letterDict["A"] = 0;
        letterDict["C"] = 1;
        letterDict["D"] = 2;
        letterDict["E"] = 3;
        letterDict["F"] = 4;
        letterDict["G"] = 5;
        letterDict["H"] = 6;
        letterDict["I"] = 7;
        letterDict["K"] = 8;
        letterDict["L"] = 9;
        letterDict["M"] = 10;
        letterDict["N"] = 11;
        letterDict["P"] = 12;
        letterDict["Q"] = 13;
        letterDict["R"] = 14;
        letterDict["S"] = 15;
        letterDict["T"] = 16;
        letterDict["V"] = 17;
        letterDict["W"] = 18;
        letterDict["Y"] = 19;
        #letterDict['Z'] = 23

        #
        #
        Matr = np.zeros((len(rawseq), window_size, ONE_HOT_SIZE))
        samplenumber = 0
        for seq in rawseq:
            AANo = 0
            #print(seq)
            for AA in seq:
                index = letterDict[AA]
                Matr[samplenumber][AANo][index] = 1
                AANo = AANo+1
            samplenumber = samplenumber + 1

    return Matr, targetY,rawseq,length

def getMatrixLabelnlp(positive_position_file_name, window_size=51, empty_aa = '*'):
    # input format   label, proteinName, postion, shortsequence
    # label存储0/1值
    #positive_position_file_name='trainingAMP.csv'
    #window_size=50
    prot = []  #
    pos = []  #
    rawseq = [] #
    all_label = [] #
    all_labela = []
    length=[]
    short_seqs = []
    #half_len = (window_size - 1) / 2

    with open(positive_position_file_name, 'r') as rf:
        reader = csv.reader(rf)
        for row in reader:

                #position = int(row[2])
                a=window_size-len(row[1])
                sseq = row[1]#+a*' '
                rawseq.append(sseq)
                b=len(row[1])
                length.append(b)
                #center = sseq[position - 1]
            # 
                all_label.append(int(row[0]))
                #all_labela.append(int(row[2])) 
                #prot.append(row[1])
                #pos.append(row[2])

        
        # Keras的utilities，用于“Converts a class vector (integers) to binary class matrix.”
        # “A binary matrix representation of the input”
        targetY = kutils.to_categorical(all_label)
        #targetYa = kutils.to_categorical(all_labela)

        ONE_HOT_SIZE = 20
        # _aminos = 'ACDEFGHIKLMNPQRSTVWY*'
        letterDict = {}
        letterDict["A"] = 0;
        letterDict["C"] = 1;
        letterDict["D"] = 2;
        letterDict["E"] = 3;
        letterDict["F"] = 4;
        letterDict["G"] = 5;
        letterDict["H"] = 6;
        letterDict["I"] = 7;
        letterDict["K"] = 8;
        letterDict["L"] = 9;
        letterDict["M"] = 10;
        letterDict["N"] = 11;
        letterDict["P"] = 12;
        letterDict["Q"] = 13;
        letterDict["R"] = 14;
        letterDict["S"] = 15;
        letterDict["T"] = 16;
        letterDict["V"] = 17;
        letterDict["W"] = 18;
        letterDict["Y"] = 19;

    return rawseq,length
# 建立网络的函数
def conv_factory(x, init_form, nb_filter, filter_size_block, dropout_rate, weight_decay):
    """Apply BatchNorm, Relu 3x3Conv2D, optional dropout

    :param x: Input keras network
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor

    :returns: keras network with b_norm, relu and convolution2d added
    :rtype: keras network
    """
    #x = Activation('relu')(x)
    # 参数的名称有修改
    x = Conv1D(nb_filter, filter_size_block,
                      kernel_initializer=init_form,
                      activation='relu',
                      padding='same',
                      use_bias=False,
                      kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition(x, init_form, nb_filter, dropout_rate, weight_decay):
    """Apply BatchNorm, Relu 1x1Conv2D, optional dropout and Maxpooling2D

    :param x: keras model
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor

    :returns: model
    :rtype: keras model, after applying batch_norm, relu-conv, dropout, maxpool

    """
    #x = Activation('relu')(x)
    x = Conv1D(nb_filter, 1,
                      kernel_initializer=init_form,
                      activation='relu',
                      padding='same',
                      use_bias=False,
                      kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    #x = AveragePooling2D((2, 2),padding='same')(x)
    x = AveragePooling1D(pool_size=5, padding='same')(x)
    #x = AveragePooling2D((2,2), padding='same')(x)
    return x

def transitionh(x, init_form, nb_filter, dropout_rate, weight_decay):
    """Apply BatchNorm, Relu 1x1Conv2D, optional dropout and Maxpooling2D

    :param x: keras model
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor

    :returns: model
    :rtype: keras model, after applying batch_norm, relu-conv, dropout, maxpool

    """
    #x = Activation('relu')(x)
    x = Conv1D(nb_filter, 1,
                      kernel_initializer=init_form,
                      activation='relu',
                      padding='same',
                      use_bias=False,
                      kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    #x = MaxPooling2D((2, 2),padding='same')(x)
    x = MaxPooling1D(pool_size=5, padding='same')(x)
    #x = MaxPooling2D((2,2), padding='same')(x)

    return x


def denseblock(x, init_form, nb_layers, nb_filter, growth_rate,filter_size_block,
               dropout_rate, weight_decay):
    """Build a denseblock where the output of each
       conv_factory is fed to subsequent ones

    :param x: keras model
    :param nb_layers: int -- the number of layers of conv_
                      factory to append to the model.
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor

    :returns: keras model with nb_layers of conv_factory appended
    :rtype: keras model

    """

    list_feat = [x]
    concat_axis = -1

    for i in range(nb_layers):
        x = conv_factory(x, init_form, growth_rate, filter_size_block, dropout_rate, weight_decay)
        #x=encoder1(x)
        list_feat.append(x)
        x = concatenate(list_feat, axis=concat_axis)
        nb_filter += growth_rate
    return x

def make_data(sentences, positive_position_file_name):
     import torch
     enc_input= []
     enc_inputs= []
     enc_input6=[]
     win1 = 50

     X1, T, rawseq, length = getMatrixLabelh(positive_position_file_name, win1)

     sentences=rawseq

     letterDict = {'A':0,'C':1,'D':1,'E':3,'F':4,'G':5,'H':6,'I':7,'K':8,'L':9,'M':10,'N':11,'P':12,'Q':13,'R':14,'S':15,'T':16,'V':17,'W':18,'Y':19,' ':20}



     src_vocab1 = letterDict   # 词源字典  字：索引
     src_idx2word = {letterDict[key]: key for key in src_vocab1}
     src_vocab_size = len(src_vocab1)
     with open(positive_position_file_name, 'r') as rf:
        reader = csv.reader(rf)
        for row in reader:
              enc_input1 = [[src_vocab1[n] for n in row[1]]]
              aaa=[21]*(50-len(row[1]))
              enc_input6.append(enc_input1[0]+aaa)
        enc_inputs.extend(enc_input6)
        return torch.LongTensor(enc_inputs)


def Phos1(nb_classes, nb_layers,img_dim1,img_dim2,init_form, nb_dense_block,
             growth_rate,filter_size_block1,filter_size_block2,filter_size_block3,filter_size_block4,filter_size_block5,filter_size_block6,
             nb_filter, filter_size_ori,dense_number, dropout_rate, dropout_dense, weight_decay):
    """ Build the DenseNet model

    :param nb_classes: int -- number of classes
    :param img_dim: tuple -- (channels, rows, columns)
    :param depth: int -- how many layers
    :param nb_dense_block: int -- number of dense blocks to add to end
    :param growth_rate: int -- number of filters to add
    :param nb_filter: int -- number of filters
    :param dropout_rate: float -- dropout rate
    :param weight_decay: float -- weight decay
    :param nb_layers:int --numbers of layers in a dense block
    :param filter_size_ori: int -- filter size of first conv1d
    :param dropout_dense: float---drop out rate of dense

    :returns: keras model with nb_layers of conv_factory appended
    :rtype: keras model

    """
    
    
    
    
    # first input of 33 seq #
    main_input = Input(shape=img_dim1)
    #import tensorflow as tf
    #main_input=encoder1(main_input)

    # Initial convolution
    x1 = Conv1D(nb_filter, filter_size_ori,
                      kernel_initializer = init_form,
                      activation='relu',
                      padding='same',
                      use_bias=False,
                      kernel_regularizer=l2(weight_decay))(main_input)#main_input
    x11 = x1
    x1 = denseblock(x1, init_form, nb_layers, nb_filter, growth_rate,filter_size_block1,
                                  dropout_rate=dropout_rate,
                                  weight_decay=weight_decay)
    xxx1 = denseblock(x1, init_form, nb_layers, nb_filter, growth_rate, filter_size_block1,
                        dropout_rate=dropout_rate,
                        weight_decay=weight_decay)
    # xxx1 = denseblock(xxx1, init_form, nb_layers, nb_filter, growth_rate, filter_size_block1,
    #                     dropout_rate=dropout_rate,
    #                     weight_decay=weight_decay)

    # second input of 21 seq #
    input2 = Input(shape=img_dim2)
    #input2=encoder1(input2)

    x2 = Conv1D(nb_filter, filter_size_ori,
                kernel_initializer = init_form,
                activation='relu',
                padding='same',
                use_bias=False,
                kernel_regularizer=l2(weight_decay))(input2)
    
    xxx2 = denseblock(x2, init_form, nb_layers, nb_filter, growth_rate, filter_size_block2,
                        dropout_rate=dropout_rate,
                        weight_decay=weight_decay)
    xxx2 = denseblock(xxx2, init_form, nb_layers, nb_filter, growth_rate, filter_size_block2,
                        dropout_rate=dropout_rate,
                        weight_decay=weight_decay)
    

    # xxx2 = denseblock(x2, init_form, nb_layers, nb_filter, growth_rate, filter_size_block2,
    #                     dropout_rate=dropout_rate,
    #                     weight_decay=weight_decay)
    #xxx2 = denseblock(xxx2, init_form, nb_layers, nb_filter, growth_rate, filter_size_block2,
                        #dropout_rate=dropout_rate,
                        #weight_decay=weight_decay)


    # The last denseblock does not have a transition


    xy = concatenate([xxx1,xxx2], axis=-1, name='contact_multi_seq')
    channel=xy.shape[-1]
    xy = layers.Dense(channel*0.25)(xy)#0.5
    xy = Activation('relu',name='seq1')(xy)
    xy = layers.Dense(channel*0.5)(xy)
    #xy = tf.nn.sigmoid(xy)
    #parameters=xy.shape
    #xy=synthetic_attention1(xy)
    #xy=attention_mechanism(xy)
    #xy,tt,al =block3(xy)
    xy = denseblock(xy, init_form, nb_layers, nb_filter, growth_rate, filter_size_block1,
                         dropout_rate=dropout_rate,
                        weight_decay=weight_decay)
    #xy=synthetic_attention1(xy)
    #xy=synthetic_attention1(xy)
    #xxx3 = Flatten()(xxx3)
    #xy=concatenate([xy,xxx3])
    xy = Flatten()(xy)

    # xxx = Dense(dense_number,
    #           name ='Dense_1',
    #           activation='relu',
    #           kernel_initializer = init_form,
    #           kernel_regularizer=l2(weight_decay),
    #           bias_regularizer=l2(weight_decay))(xy)

    xxx = Dropout(dropout_dense)(xy)
    
    #softmax
    xxx = Dense(nb_classes,
              name = 'Dense_softmax',
              activation='softmax',
              kernel_initializer = init_form,
              kernel_regularizer=l2(weight_decay),
              bias_regularizer=l2(weight_decay))(xxx)
    #xxx = Flatten()(xxx)

    
    phos_model = Model(inputs=[main_input,input2], outputs=[xxx], name="multi-DenseNet")
    #feauture_dense = Model(input=[main_input, input2, input3], output=[x], name="multi-DenseNet")

    return phos_model
#
def model_net(X_train1, X_train2, X_train3, y_train,
              nb_epoch=60,weights=None):

    nb_classes = 2
    img_dim1 = X_train1.shape[1:]
    img_dim2 = X_train2.shape[1:]
    img_dim3 = X_train3.shape[1:]

    ##########parameters#########

    init_form = 'RandomUniform'
    learning_rate = 0.001
    nb_dense_block = 1
    nb_layers = 5
    nb_filter = 32
    growth_rate = 32
    # growth_rate = 24
    filter_size_block1 = 13
    filter_size_block2 = 7
    filter_size_block3 = 3
    filter_size_ori = 1
    dense_number = 32
    dropout_rate = 0.2
    dropout_dense = 0.3
    weight_decay = 0.0001
    nb_batch_size = 512



    ###################
    # Construct model #
    ###################
    # from phosnet import Phos
    model = Phos1(nb_classes, nb_layers, img_dim1, img_dim2, img_dim3, init_form, nb_dense_block,
                             growth_rate, filter_size_block1, filter_size_block2, filter_size_block3,
                             nb_filter, filter_size_ori,
                             dense_number, dropout_rate, dropout_dense, weight_decay)
    # Model output

    # choose optimazation
    opt = adam_v2.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # model compile
    model.compile(loss='binary_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy'])

    # load weights#
    if weights is not None:
        model.load_weights(weights)
        # model2 = copy.deepcopy(model)
        model2 = model
        model2.load_weights(weights)
        for num in range(len(model2.layers) - 1):
            model.layers[num].set_weights(model2.layers[num].get_weights())

    if nb_epoch > 0 :
      model.fit([X_train1, X_train2, X_train3], y_train, batch_size=nb_batch_size,
                         # validation_data=([X_val1, X_val2, X_val3, y_val),
                         # validation_split=0.1,
                         epochs= nb_epoch, shuffle=True, verbose=1)


    return model


# 处理测试数据
def getMatrixInput(positive_position_file_name,sites, window_size=51, empty_aa = '*'):
    number=1
    a=0
    # input format  proteinName, postion, shortsequence,
    prot = []  # list of protein name
    pos = []  # list of position with protein name
    rawseq = []
    # all_label = []

    short_seqs = []
    half_len = (window_size - 1) / 2

    with open(positive_position_file_name, 'r') as rf:
        reader = csv.reader(rf)
        for row in reader:
                position = int(row[1])
                sseq = row[2]
                a=window_size-len(row[2])
                sseq = row[2]+a*' '
                rawseq.append(row[1])
                center = sseq[position - 1]
            # 
                #all_label.append(int(row[0]))
                prot.append(row[0])
                pos.append(row[1])

                #short seq
                if position - half_len > 0:
                    start = int(position - half_len)
                    left_seq = sseq[start - 1:position - 1]
                else:
                    left_seq = sseq[0:position - 1]

                end = len(sseq)
                if position + half_len < end:
                    end = int(position + half_len)
                right_seq = sseq[position:end]

                if len(left_seq) < half_len:
                    nb_lack = half_len - len(left_seq)
                    left_seq = ''.join([empty_aa for count in range(nb_lack)]) + left_seq

                if len(right_seq) < half_len:
                    nb_lack = half_len - len(right_seq)
                    right_seq = right_seq + ''.join([empty_aa for count in range(nb_lack)])
                shortseq = left_seq + center + right_seq
                short_seqs.append(shortseq)
                

        all_label = [0] *5 + [1]*(len(sseq)-5)
        targetY = kutils.to_categorical(all_label)

        ONE_HOT_SIZE = 21
        # _aminos = 'ACDEFGHIKLMNPQRSTVWY*'
        letterDict = {}
        letterDict["A"] = 0
        letterDict["C"] = 1
        letterDict["D"] = 2
        letterDict["E"] = 3
        letterDict["F"] = 4
        letterDict["G"] = 5
        letterDict["H"] = 6
        letterDict["I"] = 7
        letterDict["K"] = 8
        letterDict["L"] = 9
        letterDict["M"] = 10
        letterDict["N"] = 11
        letterDict["P"] = 12
        letterDict["Q"] = 13
        letterDict["R"] = 14
        letterDict["S"] = 15
        letterDict["T"] = 16
        letterDict["V"] = 17
        letterDict["W"] = 18
        letterDict["Y"] = 19
        letterDict[" "] = 20

        # print len(short_seqs)
        if number==1:
         Matr = np.zeros((len(rawseq), window_size, ONE_HOT_SIZE))
        else:
         a=a+1
        number=number+1
        samplenumber = 0
        for seq in short_seqs:
            AANo = 0
            for AA in seq:
                try:
                    index = letterDict[AA]
                except:
                    index = 20
                # print index
                Matr[samplenumber][AANo][index] = 1
                # print samplenumber
                AANo = AANo+1
            samplenumber = samplenumber + 1

    return Matr, targetY, prot, pos

def getMatrixInputh(positive_position_file_name,sites, window_size=51, empty_aa = '*'):
    # input format   label, proteinName, postion, shortsequence
    # label 0/1
    #positive_position_file_name='TrainingAMP.csv'
    prot = []  # 
    pos = []  # 
    rawseq = [] # 
    all_label = [] # 
    window_size=200
    short_seqs = []

    with open(positive_position_file_name, 'r') as rf:
        reader = csv.reader(rf)
        for row in reader:

                #position = int(row[2])
                a=window_size-len(row[0])
                sseq = row[0]+a*' '
                rawseq.append(sseq)
                #center = sseq[position - 1]
            # 
                #all_label.append(int(row[0]))
                #prot.append(row[1])
                #pos.append(row[2])

        
        # Keras的utilities，用于“Converts a class vector (integers) to binary class matrix.”
        # “A binary matrix representation of the input”
        #targetY = kutils.to_categorical(all_label)

        ONE_HOT_SIZE = 21
        # _aminos = 'ACDEFGHIKLMNPQRSTVWY*'
        letterDict = {}
        letterDict["A"] = 0;
        letterDict["C"] = 1;
        letterDict["D"] = 2;
        letterDict["E"] = 3;
        letterDict["F"] = 4;
        letterDict["G"] = 5;
        letterDict["H"] = 6;
        letterDict["I"] = 7;
        letterDict["K"] = 8;
        letterDict["L"] = 9;
        letterDict["M"] = 10;
        letterDict["N"] = 11;
        letterDict["P"] = 12;
        letterDict["Q"] = 13;
        letterDict["R"] = 14;
        letterDict["S"] = 15;
        letterDict["T"] = 16;
        letterDict["V"] = 17;
        letterDict["W"] = 18;
        letterDict["Y"] = 19;
        letterDict[' '] = 20;
        #letterDict['Z'] = 23

        # 
        # 
        Matr = np.zeros((len(rawseq), window_size, ONE_HOT_SIZE))
        samplenumber = 0
        for seq in rawseq:
            AANo = 0
            for AA in seq:
                index = letterDict[AA]
                Matr[samplenumber][AANo][index] = 1
                AANo = AANo+1
            samplenumber = samplenumber + 1

    return Matr

import csv
import numpy as np
import keras.utils.np_utils as kutils
# from keras.optimizers import Adam, SGD
from keras.optimizers import adam_v2
from keras.layers import Conv1D,Conv2D, MaxPooling2D
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

def getMatrixLabelFingerprint(positive_position_file_name, window_size=51, empty_aa = '*'): 
    # input format   label, proteinName, postion, shortsequence
    # label 0/1
    prot = []  # 
    pos = []  # 
    rawseq11 = [] # 
    all_label = [] # 
    window_size=91
    short_seqs = []
    #half_len = (window_size - 1) / 2
    ONE_HOT_SIZE = 17
    a=0
    b=0
    o=0
    number=1
    index=0
    positive_position_file_name='list.csv'#'common_sequences_at_least_5.csv'#'train_test1.csv'#'Antivirual.csv'#'测试集.csv'#'Non-AMPsfilter.csv'#'TrainingAMP.csv'#'需要预测的序列6.csv'#'需要预测的序列.csv'#'TrainingAMP.csv'#'测试集.csv'#'细菌素补充数据集.csv'#'TrainingAMP.csv'#'Extracellular space.csv'#'mitochondria.csv'#'Vesicle.csv'#'NonAMPtraining.csv'#'Negative50.csv'#'1.csv'#'Non-AMPsfilter.csv'#'Non-AMPsfilter.csv'#'Training1.csv'#'Non-AMPs.csv'#'验证数据集.csv'#'德国小蠊蛋白组.csv'#'美洲大蠊宏基因组AMPs未预测.csv'#'AMPs.csv'#'Training1T.csv'#'Non-AMPs.csv'#'AMPs.csv'#'Training1.csv'
    print('hello')
    with open(positive_position_file_name, 'r') as rf:
      reader = csv.reader(rf)
      for row in reader:
        #position = int(row[2])
        #sseq = row[1]
        rawseq11.append(row[1])
        #center = sseq[position - 1]
            #if center in sites:
        #.append(int(row[0]))
        index=index+1
        #print(index-1)
        #prot.append(row[1])
        #pos.append(row[2])
                # coding = one_hot_concat(shortseq)
                # all_codings.append(coding)
        
        # Keras的utilities，
        # 输出是“A binary matrix representation of the input”
        #targetY = kutils.to_categorical(all_label)
    index=0
    a=0
    b=0
    yyy=np.zeros(shape=(1547,1))
    Matr = np.zeros((len(rawseq11),window_size, ONE_HOT_SIZE))

    for index in range(0,len(rawseq11)):
      #result = GetProDes(rawseq11[index]).GetALL()
      #print(index)
      #for p in range(0,1547):
          #yyy[p][0]=list(result.values())[p]
      
      yyy[:]=np.array(list(GetProDes(rawseq11[index]).GetALL().values())[:],dtype=np.float64).reshape(-1,1)
      print(index)  
      for i in range(0,91):
          for j in range(0,17):
            Matr[index][i][j]=yyy[j+a][0]
          if a<1530:
            a=a+17
          else:
            a=a
      a=0#v1=np.array(list(GetProDes('MGMLKRIKTLM').GetALL().values())[:],dtype=np.float64).reshape(-1,1)
      #M=Matr[:360803]
      #np.save(file="amp_toxicity_measure.npy",arr=Matr) Matr=np.load(file="amp.npy")
#np.save(file="X2.npy",arr=aaa) X2=np.load(file="X2.npy")
#np.save(file="X2s.npy",arr=X_train2) X_train2=np.load(file="X2s.npy")
#np.save(file="Y.npy",arr=targetY)
    return Matr

def getMatrixLabelFingerprint1(positive_position_file_name, window_size=51, empty_aa = '*'):
    from propy.AAComposition import CalculateDipeptideComposition  # 
    # input format   label, proteinName, postion, shortsequence
    # label 0/1
    prot = []  # 
    pos = []  # 
    rawseq11 = [] # 
    all_label = [] # 
    window_size=91
    short_seqs = []
    #half_len = (window_size - 1) / 2
    ONE_HOT_SIZE = 17
    a=0
    b=0
    o=0
    number=1
    index=0
    #positive_position_file_name='mutation.csv'#'Antivirual.csv'#'测试集.csv'#'Non-AMPsfilter.csv'#'TrainingAMP.csv'#'需要预测的序列6.csv'#'需要预测的序列.csv'#'TrainingAMP.csv'#'测试集.csv'#'细菌素补充数据集.csv'#'TrainingAMP.csv'#'Extracellular space.csv'#'mitochondria.csv'#'Vesicle.csv'#'NonAMPtraining.csv'#'Negative50.csv'#'1.csv'#'Non-AMPsfilter.csv'#'Non-AMPsfilter.csv'#'Training1.csv'#'Non-AMPs.csv'#'验证数据集.csv'#'德国小蠊蛋白组.csv'#'美洲大蠊宏基因组AMPs未预测.csv'#'AMPs.csv'#'Training1T.csv'#'Non-AMPs.csv'#'AMPs.csv'#'Training1.csv'
    with open(positive_position_file_name, 'r') as rf:
      reader = csv.reader(rf)
      for row in reader:
        #position = int(row[2])
        #sseq = row[1]
        rawseq11.append(row[1])
        #center = sseq[position - 1]
            # 
            #if center in sites:
        #.append(int(row[0]))
        index=index+1
        #print(index-1)
        #prot.append(row[1])
        #pos.append(row[2])
                # coding = one_hot_concat(shortseq)
                # all_codings.append(coding)
        
        # Keras的utilities，
        # 输出是“A binary matrix representation of the input”
        #targetY = kutils.to_categorical(all_label)
    index=0
    a=0
    b=0
    yyy=np.zeros(shape=(400,1))
    Matr = np.zeros((len(rawseq11),20, 20))

    for index in range(0,len(rawseq11)):
      result = CalculateDipeptideComposition(rawseq11[index])
      print(index)
      for p in range(0,400):
          yyy[p][0]=list(result.values())[p]
      for i in range(0,20):
          for j in range(0,20):
            Matr[index][i][j]=yyy[j+a][0]
          if a<400:
            a=a+20
          else:
            a=a
      a=0
#np.save(file="X2.npy",arr=aaa) X2=np.load(file="X2.npy")
#np.save(file="X2s.npy",arr=X_train2) X_train2=np.load(file="X2s.npy")
#np.save(file="Y.npy",arr=targetY)
    return Matr

def getMatrixLabelFingerprintp(positive_position_file_name, window_size=51, empty_aa = '*'):
       # 
    # input format   label, proteinName, postion, shortsequence
    # label 0/1
    prot = []  # 
    pos = []  # 
    rawseq11 = [] # 
    all_label = [] # 
    window_size=91
    short_seqs = []
    #half_len = (window_size - 1) / 2
    ONE_HOT_SIZE = 17
    a=0
    b=0
    o=0
    number=1
    index=0
    positive_position_file_name='potential.csv'
    with open(positive_position_file_name, 'r') as rf:
      reader = csv.reader(rf)
      for row in reader:
        #position = int(row[2])
        #sseq = row[1]
        rawseq11.append(row[1])
        #center = sseq[position - 1]
            # 
            #if center in sites:
        #.append(int(row[0]))
        index=index+1
        #print(index-1)
        #prot.append(row[1])
        #pos.append(row[2])
                # coding = one_hot_concat(shortseq)
                # all_codings.append(coding)
        
        # Keras的utilities，用于“Converts a class vector (integers) to binary class matrix.”
        # 输出是“A binary matrix representation of the input”
        #targetY = kutils.to_categorical(all_label)
    index=0
    a=0
    b=0
    yyy=np.zeros(shape=(1547,1))
    Matr = np.zeros((len(rawseq11),window_size, ONE_HOT_SIZE))

    for index in range(0,len(rawseq11)):
      result = GetProDes(rawseq11[index]).GetALL()
      print(index)
      for p in range(0,1547):
          yyy[p][0]=list(result.values())[p]
      for i in range(0,91):
          for j in range(0,17):
            Matr[index][i][j]=yyy[j+a][0]
          if a<1530:
            a=a+17
          else:
            a=a
      a=0
#np.save(file="X2.npy",arr=aaa) X2=np.load(file="X2.npy")
#np.save(file="X2s.npy",arr=X_train2) X_train2=np.load(file="X2s.npy")
#np.save(file="Y.npy",arr=targetY)
    return Matr


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

def plot_ROC(labels,preds,savepath):
    """
    Args:
        labels : ground truth
        preds : model prediction
        savepath : save path 
    """
    #labels=y_train1[:,1] 
    #preds=predictions_p[:,1] #savepath='D://'
    fpr1, tpr1, threshold1 = metrics.roc_curve(labels, preds)  ###
    precision,recall,threshold1 = metrics.precision_recall_curve(labels, preds)
    roc_auc1 = metrics.auc(fpr1,tpr1)  ###计算auc的值，auc
    roc_auc1 = metrics.auc(recall,precision)
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr1, tpr1, color='darkorange',
            lw=lw, label='AUC = %0.2f' % roc_auc1)  ###
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    # plt.title('ROCs for Densenet')
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig(savepath)

#（2）空间注意力机制
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
#（1）通道注意力
def channel_attenstion(inputs, ratio=0.25):
 
    channel = inputs.shape[-1]  # 
 
    # 
    # [h,w,c]==>[None,c]
    x_max = layers.GlobalMaxPooling1D()(inputs)
    x_avg = layers.GlobalAveragePooling1D()(inputs)
 
    # [None,c]==>[1,1,c]
    x_max = layers.Reshape([1,1,-1])(x_max)  # -1
    x_avg = layers.Reshape([1,1,-1])(x_avg)  # 
 
    # 1/4, [1,1,c]==>[1,1,c//4]
    x_max = layers.Dense(channel*ratio)(x_max)
    x_avg = layers.Dense(channel*ratio)(x_avg)
 
    # relu
    x_max = layers.Activation('relu')(x_max)
    x_avg = layers.Activation('relu')(x_avg)
 
    #  [1,1,c//4]==>[1,1,c]
    x_max = layers.Dense(channel)(x_max)
    x_avg = layers.Dense(channel)(x_avg)
 
    # [1,1,c]+[1,1,c]==>[1,1,c]
    x = layers.Add()([x_max, x_avg])
 
    # 
    x = tf.nn.sigmoid(x)
 
    # 
    x = layers.Multiply()([inputs, x])  # [h,w,c]*[1,1,c]==>[h,w,c]
 
    return x



def Phos11(nb_classes, nb_layers,img_dim1,img_dim2,img_dim3,img_dim4,img_dim5,img_dim6, init_form, nb_dense_block,
             growth_rate,filter_size_block1,filter_size_block2,filter_size_block3,filter_size_block4,filter_size_block5,filter_size_block6,
             nb_filter, filter_size_ori,dense_number, dropout_rate, dropout_dense, weight_decay):
    """ Build the DenseNet model

    :param nb_classes: int -- number of classes
    :param img_dim: tuple -- (channels, rows, columns)
    :param depth: int -- how many layers
    :param nb_dense_block: int -- number of dense blocks to add to end
    :param growth_rate: int -- number of filters to add
    :param nb_filter: int -- number of filters
    :param dropout_rate: float -- dropout rate
    :param weight_decay: float -- weight decay
    :param nb_layers:int --numbers of layers in a dense block
    :param filter_size_ori: int -- filter size of first conv1d
    :param dropout_dense: float---drop out rate of dense

    :returns: keras model with nb_layers of conv_factory appended
    :rtype: keras model

    """
    
    
    
    
    # first input of 33 seq #
    main_input = Input(shape=img_dim1)
    #import tensorflow as tf
    #main_input=encoder1(main_input)
    #main_input=torch.Tensor(aaa)
    #B = 512
    #N = 50
    #C = 20
    #num_head=2
    #x = torch.rand((B, N, C))
    #MHSA=MultiHead_SelfAttention(C, num_head)
    #x1=MHSA(main_input)
    #x1=x1.detach().numpy()
    #x1 = Input(shape=img_dim1)

    # Initial convolution
    x1 = Conv1D(nb_filter, filter_size_ori,
                      kernel_initializer = init_form,
                      activation='relu',
                      padding='same',
                      use_bias=False,
                      kernel_regularizer=l2(weight_decay))(main_input)#main_input
    x11 = x1
    x1 = denseblock(x1, init_form, nb_layers, nb_filter, growth_rate,filter_size_block1,
                                  dropout_rate=dropout_rate,
                                  weight_decay=weight_decay)
    #x1=encoder1(x1)
    x11 = x1
    
    x11 = transitionh(x11, init_form, nb_filter, dropout_rate=dropout_rate,
                       weight_decay=weight_decay)
    x1 = transition(x1, init_form, nb_filter, dropout_rate=dropout_rate,
                       weight_decay=weight_decay)
    
    x01=concatenate([x1,x11])
    channel = x01.shape[-1]
    x1 = layers.Dense(channel*0.25)(x1)
    x11 = layers.Dense(channel*0.25)(x11)
    x1 = Activation('relu',name='seq1')(x1)
    x11 = Activation('relu',name='seq11')(x11)
    x1 = layers.Dense(channel)(x1)
    x11 = layers.Dense(channel)(x11)
    xxx1=layers.Add()([x1, x11])
    #xxx1=x01
    xxx1 = tf.nn.sigmoid(xxx1)
    xxx1 = layers.Multiply()([x01, xxx1])
    xxx1 = denseblock(xxx1, init_form, nb_layers, nb_filter, growth_rate, filter_size_block1,
                        dropout_rate=dropout_rate,
                        weight_decay=weight_decay)

    # The last denseblock does not have a transition

    
    

    # contact 3 output features #
    x = concatenate([xxx1], axis=-2, name='contact_multi_seq')
    #x=encoder1(x)
    #x = GlobalAveragePooling1D()(x)

    x = Flatten()(x)

    x = Dense(dense_number,
              name ='Dense_1',
              activation='relu',
              kernel_initializer = init_form,
              kernel_regularizer=l2(weight_decay),
              bias_regularizer=l2(weight_decay))(x)

    x = Dropout(dropout_dense)(x)
    
    #softmax
    x = Dense(nb_classes,
              name = 'Dense_softmax',
              activation='softmax',
              kernel_initializer = init_form,
              kernel_regularizer=l2(weight_decay),
              bias_regularizer=l2(weight_decay))(x)
    #x = Flatten()(x)

    
    phos_model = Model(inputs=[main_input], outputs=[x], name="multi-DenseNet")
    #feauture_dense = Model(input=[main_input, input2, input3], output=[x], name="multi-DenseNet")

    return phos_model




def PhosG(nb_classes, nb_layers,img_dim1,img_dim2,img_dim3,img_dim4,img_dim5,img_dim6, init_form, nb_dense_block,
             growth_rate,filter_size_block1,filter_size_block2,filter_size_block3,filter_size_block4,filter_size_block5,filter_size_block6,
             nb_filter, filter_size_ori,dense_number, dropout_rate, dropout_dense, weight_decay):
    """ Build the DenseNet model

    :param nb_classes: int -- number of classes
    :param img_dim: tuple -- (channels, rows, columns)
    :param depth: int -- how many layers
    :param nb_dense_block: int -- number of dense blocks to add to end
    :param growth_rate: int -- number of filters to add
    :param nb_filter: int -- number of filters
    :param dropout_rate: float -- dropout rate
    :param weight_decay: float -- weight decay
    :param nb_layers:int --numbers of layers in a dense block
    :param filter_size_ori: int -- filter size of first conv1d
    :param dropout_dense: float---drop out rate of dense

    :returns: keras model with nb_layers of conv_factory appended
    :rtype: keras model

    """
    
    
    
    
    # first input of 33 seq #
    main_input = Input(shape=img_dim1)


    # Initial convolution
    x1 = Conv1D(nb_filter, filter_size_ori,
                      kernel_initializer = init_form,
                      activation='relu',
                      padding='same',
                      use_bias=False,
                      kernel_regularizer=l2(weight_decay))(main_input)#main_input
    x11 = x1
    x1 = denseblock(x1, init_form, nb_layers, nb_filter, growth_rate,filter_size_block1,
                                  dropout_rate=dropout_rate,
                                  weight_decay=weight_decay)
    

    #x1=encoder1(x1)
    x11 = x1
    #x1=non_local_attention(x1)
    #x11=encoder1(x11)

    x11 = transitionh(x11, init_form, nb_filter, dropout_rate=dropout_rate,
                       weight_decay=weight_decay)
    x1 = transition(x1, init_form, nb_filter, dropout_rate=dropout_rate,
                       weight_decay=weight_decay)
    #x1=non_local_attention(x1)
    #x11=encoder1(x11)
   
    x01=concatenate([x1,x11])
    channel = x01.shape[-1]
    x1 = layers.Dense(channel*0.25)(x1)
    x11 = layers.Dense(channel*0.25)(x11)
    x1 = Activation('relu',name='seq1')(x1)
    x11 = Activation('relu',name='seq11')(x11)
    x1 = layers.Dense(channel)(x1)
    x11 = layers.Dense(channel)(x11)
    xxx1=layers.Add()([x1, x11])
    xxx1 = tf.nn.sigmoid(xxx1)
    #xxx1=encoder1(xxx1)
    xxx1 = layers.Multiply()([x01, xxx1])
    xxx1 = denseblock(xxx1, init_form, nb_layers, nb_filter, growth_rate, filter_size_block1,
                        dropout_rate=dropout_rate,
                        weight_decay=weight_decay)

    # second input of 21 seq #
    input2 = Input(shape=img_dim2)
    #input2=encoder1(input2)

    x2 = Conv1D(nb_filter, filter_size_ori,
                kernel_initializer = init_form,
                activation='relu',
                padding='same',
                use_bias=False,
                kernel_regularizer=l2(weight_decay))(input2)
    
    x2 = denseblock(x2, init_form, nb_layers, nb_filter, growth_rate, filter_size_block2,
                        dropout_rate=dropout_rate,
                        weight_decay=weight_decay)

    
    
    x22 = x2  

    x22 = transitionh(x22, init_form, nb_filter, dropout_rate=dropout_rate,
                        weight_decay=weight_decay)
    x2 = transition(x2, init_form, nb_filter, dropout_rate=dropout_rate,
                        weight_decay=weight_decay)
    #x2=non_local_attention(x2)
    #x22=encoder1(x22)

    x02=concatenate([x2,x22])
    channel = x02.shape[-1]
    x2 = layers.Dense(channel*0.25)(x2)
    x22 = layers.Dense(channel*0.25)(x22)
    x2 = Activation('relu',name='seq2')(x2)
    x22 = Activation('relu',name='seq22')(x22)
    x2 = layers.Dense(channel)(x2)
    x22 = layers.Dense(channel)(x22)
    xxx2=layers.Add()([x2, x22])
    xxx2 = tf.nn.sigmoid(xxx2)
    #xxx2=encoder1(xxx2)
    xxx2 = layers.Multiply()([x02, xxx2])
    xxx2 = denseblock(xxx2, init_form, nb_layers, nb_filter, growth_rate, filter_size_block2,
                        dropout_rate=dropout_rate,
                        weight_decay=weight_decay)
    #xxx2 = denseblock(xxx2, init_form, nb_layers, nb_filter, growth_rate, filter_size_block2,
                        #dropout_rate=dropout_rate,
                        #weight_decay=weight_decay)


    # The last denseblock does not have a transition

    #third input seq3 of 15 #
    input3 = Input(shape=img_dim3)

    x3 = Conv1D(nb_filter, filter_size_ori,
                kernel_initializer = init_form,
                activation='relu',
                padding='same',
                use_bias=False,
                kernel_regularizer=l2(weight_decay))(input3)
    x33 = x3
    
    x3 = denseblock(x3, init_form, nb_layers, nb_filter, growth_rate, filter_size_block3,
                        dropout_rate=dropout_rate,
                        weight_decay=weight_decay)

    

    x33 = x3
    #x3=non_local_attention(x3)
    #x33=encoder1(x33)
    
    x33 = transition(x33, init_form, nb_filter, dropout_rate=dropout_rate,
                       weight_decay=weight_decay)
    x3 = transition(x3, init_form, nb_filter, dropout_rate=dropout_rate,
                       weight_decay=weight_decay)
    #x3=non_local_attention(x3)
    #x33=encoder1(x33)               
    
    x03=concatenate([x3,x33])
    channel = x03.shape[-1]
    x3 = layers.Dense(channel*0.25)(x3)
    x33 = layers.Dense(channel*0.25)(x33)
    x3 = Activation('relu',name='seq13')(x3)
    x33 = Activation('relu',name='seq133')(x33)
    x3 = layers.Dense(channel)(x3)
    x33 = layers.Dense(channel)(x33)
    xxx3=layers.Add()([x3, x33])
    xxx3 = tf.nn.sigmoid(xxx3)
    #xxx3=encoder1(xxx3)
    xxx3 = layers.Multiply()([x03, xxx3])
    
    xxx3 = denseblock(xxx3, init_form, nb_layers, nb_filter, growth_rate, filter_size_block3,
                        dropout_rate=dropout_rate,
                        weight_decay=weight_decay)

      
    # contact 3 output features #
    #x16 = concatenate([xxx1,xxx2,xxx3], axis=-2, name='contact_multi_seq')
    #channel1 = x16.shape[-1]
    #x16 = layers.Dense(channel1*0.25)(x16)
    #x16 = Activation('relu',name='seq666')(x16)
    #x16 = layers.Dense(channel)(x16)
    #x16 = tf.nn.sigmoid(x16)
    #xxx=encoder1(x16)
    #xy=non_local_attention(xy)
    #xy=synthetic_attention1(xy)
    #xxx1 = cbam_block(xxx1)
    #xxx2 = cbam_block(xxx2)
    #xxx3 = cbam_block(xxx3)
    xy = concatenate([xxx1,xxx2,xxx3], axis=-2, name='contact_multi_seq')
    #xy=synthetic_attention1(xy)
    #xy=attention_mechanism(xy)
    xy,tt=cbam_block(xy)
    #xy = denseblock(xy, init_form, nb_layers, nb_filter, growth_rate, filter_size_block1,
                        #dropout_rate=dropout_rate,
                        #weight_decay=weight_decay)
    #xy=synthetic_attention1(xy)
    #xy=synthetic_attention1(xy)
    #xxx3 = Flatten()(xxx3)
    #xy=concatenate([xy,xxx3])
    xy = Flatten()(xy)

    xxx = Dense(dense_number,
              name ='Dense_1',
              activation='relu',
              kernel_initializer = init_form,
              kernel_regularizer=l2(weight_decay),
              bias_regularizer=l2(weight_decay))(xy)

    xxx = Dropout(dropout_dense)(xxx)
    
    #softmax
    xxx = Dense(nb_classes,
              name = 'Dense_softmax',
              activation='softmax',
              kernel_initializer = init_form,
              kernel_regularizer=l2(weight_decay),
              bias_regularizer=l2(weight_decay))(xxx)
    #xxx = Flatten()(xxx)

    
    phos_model = Model(inputs=[main_input,input2,input3], outputs=[xxx], name="multi-DenseNet")
    #feauture_dense = Model(input=[main_input, input2, input3], output=[x], name="multi-DenseNet")

    return phos_model,tt

def PhosF(nb_classes, nb_layers,img_dim1,img_dim2,img_dim3,img_dim4,img_dim5,img_dim6, init_form, nb_dense_block,
             growth_rate,filter_size_block1,filter_size_block2,filter_size_block3,filter_size_block4,filter_size_block5,filter_size_block6,
             nb_filter, filter_size_ori,dense_number, dropout_rate, dropout_dense, weight_decay):
    """ Build the DenseNet model

    :param nb_classes: int -- number of classes
    :param img_dim: tuple -- (channels, rows, columns)
    :param depth: int -- how many layers
    :param nb_dense_block: int -- number of dense blocks to add to end
    :param growth_rate: int -- number of filters to add
    :param nb_filter: int -- number of filters
    :param dropout_rate: float -- dropout rate
    :param weight_decay: float -- weight decay
    :param nb_layers:int --numbers of layers in a dense block
    :param filter_size_ori: int -- filter size of first conv1d
    :param dropout_dense: float---drop out rate of dense

    :returns: keras model with nb_layers of conv_factory appended
    :rtype: keras model

    """
    
    
    
    
    # first input of 33 seq #
    main_input = Input(shape=img_dim1)
    #import tensorflow as tf
    main_input=encoder1(main_input)

    # Initial convolution
    x1 = Conv1D(nb_filter, filter_size_ori,
                      kernel_initializer = init_form,
                      activation='relu',
                      padding='same',
                      use_bias=False,
                      kernel_regularizer=l2(weight_decay))(main_input)#main_input
    x11 = x1
    x1 = denseblock(x1, init_form, nb_layers, nb_filter, growth_rate,filter_size_block1,
                                  dropout_rate=dropout_rate,
                                  weight_decay=weight_decay)
    

    #x1=encoder1(x1)
    x11 = x1
    #x1=non_local_attention(x1)
    #x11=encoder1(x11)

    x11 = transitionh(x11, init_form, nb_filter, dropout_rate=dropout_rate,
                       weight_decay=weight_decay)
    x1 = transition(x1, init_form, nb_filter, dropout_rate=dropout_rate,
                       weight_decay=weight_decay)
    #x1=non_local_attention(x1)
    #x11=encoder1(x11)
   
    x01=concatenate([x1,x11])
    channel = x01.shape[-1]
    x1 = layers.Dense(channel*0.25)(x1)
    x11 = layers.Dense(channel*0.25)(x11)
    x1 = Activation('relu',name='seq1')(x1)
    x11 = Activation('relu',name='seq11')(x11)
    x1 = layers.Dense(channel)(x1)
    x11 = layers.Dense(channel)(x11)
    xxx1=layers.Add()([x1, x11])
    xxx1 = tf.nn.sigmoid(xxx1)
    #xxx1=encoder1(xxx1)
    xxx1 = layers.Multiply()([x01, xxx1])
    xxx1 = denseblock(xxx1, init_form, nb_layers, nb_filter, growth_rate, filter_size_block1,
                        dropout_rate=dropout_rate,
                        weight_decay=weight_decay)

    # second input of 21 seq #
    input2 = Input(shape=img_dim2)
    input2=encoder1(input2)

    x2 = Conv1D(nb_filter, filter_size_ori,
                kernel_initializer = init_form,
                activation='relu',
                padding='same',
                use_bias=False,
                kernel_regularizer=l2(weight_decay))(input2)
    
    x2 = denseblock(x2, init_form, nb_layers, nb_filter, growth_rate, filter_size_block2,
                        dropout_rate=dropout_rate,
                        weight_decay=weight_decay)

    
    
    x22 = x2  
    #x2=non_local_attention(x2)
    #x22=encoder1(x22)
    x22 = transitionh(x22, init_form, nb_filter, dropout_rate=dropout_rate,
                        weight_decay=weight_decay)
    x2 = transition(x2, init_form, nb_filter, dropout_rate=dropout_rate,
                        weight_decay=weight_decay)
    #x2=non_local_attention(x2)
    #x22=encoder1(x22)

    x02=concatenate([x2,x22])
    channel = x02.shape[-1]
    x2 = layers.Dense(channel*0.25)(x2)
    x22 = layers.Dense(channel*0.25)(x22)
    x2 = Activation('relu',name='seq2')(x2)
    x22 = Activation('relu',name='seq22')(x22)
    x2 = layers.Dense(channel)(x2)
    x22 = layers.Dense(channel)(x22)
    xxx2=layers.Add()([x2, x22])
    xxx2 = tf.nn.sigmoid(xxx2)
    #xxx2=encoder1(xxx2)
    xxx2 = layers.Multiply()([x02, xxx2])
    xxx2 = denseblock(xxx2, init_form, nb_layers, nb_filter, growth_rate, filter_size_block2,
                        dropout_rate=dropout_rate,
                        weight_decay=weight_decay)
    #xxx2 = denseblock(xxx2, init_form, nb_layers, nb_filter, growth_rate, filter_size_block2,
                        #dropout_rate=dropout_rate,
                        #weight_decay=weight_decay)


    # The last denseblock does not have a transition

    #third input seq3 of 15 #
    input3 = Input(shape=img_dim3)
    input3=encoder1(input3)
    x3 = Conv1D(nb_filter, filter_size_ori,
                kernel_initializer = init_form,
                activation='relu',
                padding='same',
                use_bias=False,
                kernel_regularizer=l2(weight_decay))(input3)
    x33 = x3
    
    x3 = denseblock(x3, init_form, nb_layers, nb_filter, growth_rate, filter_size_block3,
                        dropout_rate=dropout_rate,
                        weight_decay=weight_decay)

    

    x33 = x3
    #x3=non_local_attention(x3)
    #x33=encoder1(x33)
    
    x33 = transition(x33, init_form, nb_filter, dropout_rate=dropout_rate,
                       weight_decay=weight_decay)
    x3 = transition(x3, init_form, nb_filter, dropout_rate=dropout_rate,
                       weight_decay=weight_decay)
    #x3=non_local_attention(x3)
    #x33=encoder1(x33)               
    
    x03=concatenate([x3,x33])
    channel = x03.shape[-1]
    x3 = layers.Dense(channel*0.25)(x3)
    x33 = layers.Dense(channel*0.25)(x33)
    x3 = Activation('relu',name='seq13')(x3)
    x33 = Activation('relu',name='seq133')(x33)
    x3 = layers.Dense(channel)(x3)
    x33 = layers.Dense(channel)(x33)
    xxx3=layers.Add()([x3, x33])
    xxx3 = tf.nn.sigmoid(xxx3)
    #xxx3=encoder1(xxx3)
    xxx3 = layers.Multiply()([x03, xxx3])
    
    xxx3 = denseblock(xxx3, init_form, nb_layers, nb_filter, growth_rate, filter_size_block3,
                        dropout_rate=dropout_rate,
                        weight_decay=weight_decay)

      
    # contact 3 output features #
    #x16 = concatenate([xxx1,xxx2,xxx3], axis=-2, name='contact_multi_seq')
    #channel1 = x16.shape[-1]
    #x16 = layers.Dense(channel1*0.25)(x16)
    #x16 = Activation('relu',name='seq666')(x16)
    #x16 = layers.Dense(channel)(x16)
    #x16 = tf.nn.sigmoid(x16)
    #xxx=encoder1(x16)
    #xy=non_local_attention(xy)
    #xy=synthetic_attention1(xy)
    #xxx1 = cbam_block(xxx1)
    #xxx2 = cbam_block(xxx2)
    #xxx3 = cbam_block(xxx3)
    xy = concatenate([xxx1,xxx2,xxx3], axis=-2, name='contact_multi_seq')
    #xy=synthetic_attention1(xy)
    #xy=attention_mechanism(xy)
    #xy,tt =cbam_block(xy)
    #xy = denseblock(xy, init_form, nb_layers, nb_filter, growth_rate, filter_size_block1,
                        #dropout_rate=dropout_rate,
                        #weight_decay=weight_decay)
    #xy=synthetic_attention1(xy)
    #xy=synthetic_attention1(xy)
    #xxx3 = Flatten()(xxx3)
    #xy=concatenate([xy,xxx3])
    xy = Flatten()(xy)

    xxx = Dense(dense_number,
              name ='Dense_1',
              activation='relu',
              kernel_initializer = init_form,
              kernel_regularizer=l2(weight_decay),
              bias_regularizer=l2(weight_decay))(xy)

    xxx = Dropout(dropout_dense)(xxx)
    
    #softmax
    xxx = Dense(nb_classes,
              name = 'Dense_softmax',
              activation='softmax',
              kernel_initializer = init_form,
              kernel_regularizer=l2(weight_decay),
              bias_regularizer=l2(weight_decay))(xxx)
    #xxx = Flatten()(xxx)

    
    phos_model = Model(inputs=[main_input,input2,input3], outputs=[xxx], name="multi-DenseNet")
    #feauture_dense = Model(input=[main_input, input2, input3], output=[x], name="multi-DenseNet")

    return phos_model

def PhosE(nb_classes, nb_layers,img_dim1,img_dim2,img_dim3,img_dim4,img_dim5,img_dim6, init_form, nb_dense_block,
             growth_rate,filter_size_block1,filter_size_block2,filter_size_block3,filter_size_block4,filter_size_block5,filter_size_block6,
             nb_filter, filter_size_ori,dense_number, dropout_rate, dropout_dense, weight_decay):
    """ Build the DenseNet model

    :param nb_classes: int -- number of classes
    :param img_dim: tuple -- (channels, rows, columns)
    :param depth: int -- how many layers
    :param nb_dense_block: int -- number of dense blocks to add to end
    :param growth_rate: int -- number of filters to add
    :param nb_filter: int -- number of filters
    :param dropout_rate: float -- dropout rate
    :param weight_decay: float -- weight decay
    :param nb_layers:int --numbers of layers in a dense block
    :param filter_size_ori: int -- filter size of first conv1d
    :param dropout_dense: float---drop out rate of dense

    :returns: keras model with nb_layers of conv_factory appended
    :rtype: keras model

    """
    
    
    
    
    # first input of 33 seq #
    main_input = Input(shape=img_dim1)
    #import tensorflow as tf
    main_input=encoder1(main_input)

    # Initial convolution
    x1 = Conv1D(nb_filter, filter_size_ori,
                      kernel_initializer = init_form,
                      activation='relu',
                      padding='same',
                      use_bias=False,
                      kernel_regularizer=l2(weight_decay))(main_input)#main_input
    x11 = x1
    x1 = denseblock(x1, init_form, nb_layers, nb_filter, growth_rate,filter_size_block1,
                                  dropout_rate=dropout_rate,
                                  weight_decay=weight_decay)
    

    #x1=encoder1(x1)

    xxx1 = denseblock(x1, init_form, nb_layers, nb_filter, growth_rate, filter_size_block1,
                        dropout_rate=dropout_rate,
                        weight_decay=weight_decay)

    # second input of 21 seq #
    input2 = Input(shape=img_dim2)
    input2=encoder1(input2)

    x2 = Conv1D(nb_filter, filter_size_ori,
                kernel_initializer = init_form,
                activation='relu',
                padding='same',
                use_bias=False,
                kernel_regularizer=l2(weight_decay))(input2)
    
    x2 = denseblock(x2, init_form, nb_layers, nb_filter, growth_rate, filter_size_block2,
                        dropout_rate=dropout_rate,
                        weight_decay=weight_decay)

    
    
    xxx2 = denseblock(x2, init_form, nb_layers, nb_filter, growth_rate, filter_size_block2,
                        dropout_rate=dropout_rate,
                        weight_decay=weight_decay)
    #xxx2 = denseblock(xxx2, init_form, nb_layers, nb_filter, growth_rate, filter_size_block2,
                        #dropout_rate=dropout_rate,
                        #weight_decay=weight_decay)


    # The last denseblock does not have a transition

    #third input seq3 of 15 #
    input3 = Input(shape=img_dim3)
    input3=encoder1(input3)
    x3 = Conv1D(nb_filter, filter_size_ori,
                kernel_initializer = init_form,
                activation='relu',
                padding='same',
                use_bias=False,
                kernel_regularizer=l2(weight_decay))(input3)
    x33 = x3
    
    x3 = denseblock(x3, init_form, nb_layers, nb_filter, growth_rate, filter_size_block3,
                        dropout_rate=dropout_rate,
                        weight_decay=weight_decay)

    

    
    xxx3 = denseblock(x3, init_form, nb_layers, nb_filter, growth_rate, filter_size_block3,
                        dropout_rate=dropout_rate,
                        weight_decay=weight_decay)

      
    # contact 3 output features #
    #x16 = concatenate([xxx1,xxx2,xxx3], axis=-2, name='contact_multi_seq')
    #channel1 = x16.shape[-1]
    #x16 = layers.Dense(channel1*0.25)(x16)
    #x16 = Activation('relu',name='seq666')(x16)
    #x16 = layers.Dense(channel)(x16)
    #x16 = tf.nn.sigmoid(x16)
    #xxx=encoder1(x16)
    #xy=non_local_attention(xy)
    #xy=synthetic_attention1(xy)
    #xxx1 = cbam_block(xxx1)
    #xxx2 = cbam_block(xxx2)
    #xxx3 = cbam_block(xxx3)
    xy = concatenate([xxx1,xxx2,xxx3], axis=-2, name='contact_multi_seq')
    #xy=synthetic_attention1(xy)
    #xy=attention_mechanism(xy)
    xy,tt =cbam_block(xy)
    #xy = denseblock(xy, init_form, nb_layers, nb_filter, growth_rate, filter_size_block1,
                        #dropout_rate=dropout_rate,
                        #weight_decay=weight_decay)
    #xy=synthetic_attention1(xy)
    #xy=synthetic_attention1(xy)
    #xxx3 = Flatten()(xxx3)
    #xy=concatenate([xy,xxx3])
    xy = Flatten()(xy)

    xxx = Dense(dense_number,
              name ='Dense_1',
              activation='relu',
              kernel_initializer = init_form,
              kernel_regularizer=l2(weight_decay),
              bias_regularizer=l2(weight_decay))(xy)

    xxx = Dropout(dropout_dense)(xxx)
    
    #softmax
    xxx = Dense(nb_classes,
              name = 'Dense_softmax',
              activation='softmax',
              kernel_initializer = init_form,
              kernel_regularizer=l2(weight_decay),
              bias_regularizer=l2(weight_decay))(xxx)
    #xxx = Flatten()(xxx)

    
    phos_model = Model(inputs=[main_input,input2,input3], outputs=[xxx], name="multi-DenseNet")
    #feauture_dense = Model(input=[main_input, input2, input3], output=[x], name="multi-DenseNet")

    return phos_model,tt






def PhosD(nb_classes, nb_layers,img_dim1,img_dim2,img_dim3,img_dim4,img_dim5,img_dim6, init_form, nb_dense_block,
             growth_rate,filter_size_block1,filter_size_block2,filter_size_block3,filter_size_block4,filter_size_block5,filter_size_block6,
             nb_filter, filter_size_ori,dense_number, dropout_rate, dropout_dense, weight_decay):
    """ Build the DenseNet model

    :param nb_classes: int -- number of classes
    :param img_dim: tuple -- (channels, rows, columns)
    :param depth: int -- how many layers
    :param nb_dense_block: int -- number of dense blocks to add to end
    :param growth_rate: int -- number of filters to add
    :param nb_filter: int -- number of filters
    :param dropout_rate: float -- dropout rate
    :param weight_decay: float -- weight decay
    :param nb_layers:int --numbers of layers in a dense block
    :param filter_size_ori: int -- filter size of first conv1d
    :param dropout_dense: float---drop out rate of dense

    :returns: keras model with nb_layers of conv_factory appended
    :rtype: keras model

    """
    
    
    
    
    # first input of 33 seq #
    main_input = Input(shape=img_dim1)
    #import tensorflow as tf
    main_input=encoder1(main_input)

    # Initial convolution
    x1 = Conv1D(nb_filter, filter_size_ori,
                      kernel_initializer = init_form,
                      activation='relu',
                      padding='same',
                      use_bias=False,
                      kernel_regularizer=l2(weight_decay))(main_input)#main_input
    x11 = x1
    x1 = denseblock(x1, init_form, nb_layers, nb_filter, growth_rate,filter_size_block1,
                                  dropout_rate=dropout_rate,
                                  weight_decay=weight_decay)
    

    #x1=encoder1(x1)
    #x11 = x1
    #x1=non_local_attention(x1)
    #x11=encoder1(x11)

    xxx1 = denseblock(x1, init_form, nb_layers, nb_filter, growth_rate, filter_size_block1,
                        dropout_rate=dropout_rate,
                        weight_decay=weight_decay)

    # second input of 21 seq #
    input2 = Input(shape=img_dim2)
    input2=encoder1(input2)

    x2 = Conv1D(nb_filter, filter_size_ori,
                kernel_initializer = init_form,
                activation='relu',
                padding='same',
                use_bias=False,
                kernel_regularizer=l2(weight_decay))(input2)
    
    x2 = denseblock(x2, init_form, nb_layers, nb_filter, growth_rate, filter_size_block2,
                        dropout_rate=dropout_rate,
                        weight_decay=weight_decay)

    
    

    xxx2 = denseblock(x2, init_form, nb_layers, nb_filter, growth_rate, filter_size_block2,
                        dropout_rate=dropout_rate,
                        weight_decay=weight_decay)
    #xxx2 = denseblock(xxx2, init_form, nb_layers, nb_filter, growth_rate, filter_size_block2,
                        #dropout_rate=dropout_rate,
                        #weight_decay=weight_decay)


    # The last denseblock does not have a transition

    #third input seq3 of 15 #
    input3 = Input(shape=img_dim3)
    input3=encoder1(input3)
    x3 = Conv1D(nb_filter, filter_size_ori,
                kernel_initializer = init_form,
                activation='relu',
                padding='same',
                use_bias=False,
                kernel_regularizer=l2(weight_decay))(input3)
    x33 = x3
    
    x3 = denseblock(x3, init_form, nb_layers, nb_filter, growth_rate, filter_size_block3,
                        dropout_rate=dropout_rate,
                        weight_decay=weight_decay)

    


    
    xxx3 = denseblock(x3, init_form, nb_layers, nb_filter, growth_rate, filter_size_block3,
                        dropout_rate=dropout_rate,
                        weight_decay=weight_decay)

      
    # contact 3 output features #
    #x16 = concatenate([xxx1,xxx2,xxx3], axis=-2, name='contact_multi_seq')
    #channel1 = x16.shape[-1]
    #x16 = layers.Dense(channel1*0.25)(x16)
    #x16 = Activation('relu',name='seq666')(x16)
    #x16 = layers.Dense(channel)(x16)
    #x16 = tf.nn.sigmoid(x16)
    #xxx=encoder1(x16)
    #xy=non_local_attention(xy)
    #xy=synthetic_attention1(xy)
    #xxx1 = cbam_block(xxx1)
    #xxx2 = cbam_block(xxx2)
    #xxx3 = cbam_block(xxx3)
    xy = concatenate([xxx1,xxx2,xxx3], axis=-2, name='contact_multi_seq')
    #xy=synthetic_attention1(xy)
    #xy=attention_mechanism(xy)
    #xy,tt =cbam_block(xy)
    #xy = denseblock(xy, init_form, nb_layers, nb_filter, growth_rate, filter_size_block1,
                        #dropout_rate=dropout_rate,
                        #weight_decay=weight_decay)
    #xy=synthetic_attention1(xy)
    #xy=synthetic_attention1(xy)
    #xxx3 = Flatten()(xxx3)
    #xy=concatenate([xy,xxx3])
    xy = Flatten()(xy)

    xxx = Dense(dense_number,
              name ='Dense_1',
              activation='relu',
              kernel_initializer = init_form,
              kernel_regularizer=l2(weight_decay),
              bias_regularizer=l2(weight_decay))(xy)

    xxx = Dropout(dropout_dense)(xxx)
    
    #softmax
    xxx = Dense(nb_classes,
              name = 'Dense_softmax',
              activation='softmax',
              kernel_initializer = init_form,
              kernel_regularizer=l2(weight_decay),
              bias_regularizer=l2(weight_decay))(xxx)
    #xxx = Flatten()(xxx)

    
    phos_model = Model(inputs=[main_input,input2,input3], outputs=[xxx], name="multi-DenseNet")
    #feauture_dense = Model(input=[main_input, input2, input3], output=[x], name="multi-DenseNet")

    return phos_model

def PhosC(nb_classes, nb_layers,img_dim1,img_dim2,img_dim3,img_dim4,img_dim5,img_dim6, init_form, nb_dense_block,
             growth_rate,filter_size_block1,filter_size_block2,filter_size_block3,filter_size_block4,filter_size_block5,filter_size_block6,
             nb_filter, filter_size_ori,dense_number, dropout_rate, dropout_dense, weight_decay):
    """ Build the DenseNet model

    :param nb_classes: int -- number of classes
    :param img_dim: tuple -- (channels, rows, columns)
    :param depth: int -- how many layers
    :param nb_dense_block: int -- number of dense blocks to add to end
    :param growth_rate: int -- number of filters to add
    :param nb_filter: int -- number of filters
    :param dropout_rate: float -- dropout rate
    :param weight_decay: float -- weight decay
    :param nb_layers:int --numbers of layers in a dense block
    :param filter_size_ori: int -- filter size of first conv1d
    :param dropout_dense: float---drop out rate of dense

    :returns: keras model with nb_layers of conv_factory appended
    :rtype: keras model

    """
    
    
    
    
    # first input of 33 seq #
    main_input = Input(shape=img_dim1)
    #import tensorflow as tf
    #main_input=encoder1(main_input)

    # Initial convolution
    x1 = Conv1D(nb_filter, filter_size_ori,
                      kernel_initializer = init_form,
                      activation='relu',
                      padding='same',
                      use_bias=False,
                      kernel_regularizer=l2(weight_decay))(main_input)#main_input
    x11 = x1
    x1 = denseblock(x1, init_form, nb_layers, nb_filter, growth_rate,filter_size_block1,
                                  dropout_rate=dropout_rate,
                                  weight_decay=weight_decay)
    

    #x1=encoder1(x1)
    xxx1 = denseblock(x1, init_form, nb_layers, nb_filter, growth_rate, filter_size_block1,
                        dropout_rate=dropout_rate,
                        weight_decay=weight_decay)

    # second input of 21 seq #
    input2 = Input(shape=img_dim2)
    #input2=encoder1(input2)

    x2 = Conv1D(nb_filter, filter_size_ori,
                kernel_initializer = init_form,
                activation='relu',
                padding='same',
                use_bias=False,
                kernel_regularizer=l2(weight_decay))(input2)
    
    x2 = denseblock(x2, init_form, nb_layers, nb_filter, growth_rate, filter_size_block2,
                        dropout_rate=dropout_rate,
                        weight_decay=weight_decay)

    
    
    xxx2 = denseblock(x2, init_form, nb_layers, nb_filter, growth_rate, filter_size_block2,
                        dropout_rate=dropout_rate,
                        weight_decay=weight_decay)
    #xxx2 = denseblock(xxx2, init_form, nb_layers, nb_filter, growth_rate, filter_size_block2,
                        #dropout_rate=dropout_rate,
                        #weight_decay=weight_decay)


    # The last denseblock does not have a transition

    #third input seq3 of 15 #
    input3 = Input(shape=img_dim3)
    #input3=encoder1(input3)
    x3 = Conv1D(nb_filter, filter_size_ori,
                kernel_initializer = init_form,
                activation='relu',
                padding='same',
                use_bias=False,
                kernel_regularizer=l2(weight_decay))(input3)
    x33 = x3
    
    x3 = denseblock(x3, init_form, nb_layers, nb_filter, growth_rate, filter_size_block3,
                        dropout_rate=dropout_rate,
                        weight_decay=weight_decay)

    

    
    xxx3 = denseblock(x3, init_form, nb_layers, nb_filter, growth_rate, filter_size_block3,
                        dropout_rate=dropout_rate,
                        weight_decay=weight_decay)

      
    # contact 3 output features #
    #x16 = concatenate([xxx1,xxx2,xxx3], axis=-2, name='contact_multi_seq')
    #channel1 = x16.shape[-1]
    #x16 = layers.Dense(channel1*0.25)(x16)
    #x16 = Activation('relu',name='seq666')(x16)
    #x16 = layers.Dense(channel)(x16)
    #x16 = tf.nn.sigmoid(x16)
    #xxx=encoder1(x16)
    #xy=non_local_attention(xy)
    #xy=synthetic_attention1(xy)
    #xxx1 = cbam_block(xxx1)
    #xxx2 = cbam_block(xxx2)
    #xxx3 = cbam_block(xxx3)
    xy = concatenate([xxx1,xxx2,xxx3], axis=-2, name='contact_multi_seq')
    #xy=synthetic_attention1(xy)
    #xy=attention_mechanism(xy)
    xy,tt =cbam_block(xy)
    #xy = denseblock(xy, init_form, nb_layers, nb_filter, growth_rate, filter_size_block1,
                        #dropout_rate=dropout_rate,
                        #weight_decay=weight_decay)
    #xy=synthetic_attention1(xy)
    #xy=synthetic_attention1(xy)
    #xxx3 = Flatten()(xxx3)
    #xy=concatenate([xy,xxx3])
    xy = Flatten()(xy)

    xxx = Dense(dense_number,
              name ='Dense_1',
              activation='relu',
              kernel_initializer = init_form,
              kernel_regularizer=l2(weight_decay),
              bias_regularizer=l2(weight_decay))(xy)

    xxx = Dropout(dropout_dense)(xxx)
    
    #softmax
    xxx = Dense(nb_classes,
              name = 'Dense_softmax',
              activation='softmax',
              kernel_initializer = init_form,
              kernel_regularizer=l2(weight_decay),
              bias_regularizer=l2(weight_decay))(xxx)
    #xxx = Flatten()(xxx)

    
    phos_model = Model(inputs=[main_input,input2,input3], outputs=[xxx], name="multi-DenseNet")
    #feauture_dense = Model(input=[main_input, input2, input3], output=[x], name="multi-DenseNet")

    return phos_model,tt

def PhosB(nb_classes, nb_layers,img_dim1,img_dim2,img_dim3,img_dim4,img_dim5,img_dim6, init_form, nb_dense_block,
             growth_rate,filter_size_block1,filter_size_block2,filter_size_block3,filter_size_block4,filter_size_block5,filter_size_block6,
             nb_filter, filter_size_ori,dense_number, dropout_rate, dropout_dense, weight_decay):
    """ Build the DenseNet model

    :param nb_classes: int -- number of classes
    :param img_dim: tuple -- (channels, rows, columns)
    :param depth: int -- how many layers
    :param nb_dense_block: int -- number of dense blocks to add to end
    :param growth_rate: int -- number of filters to add
    :param nb_filter: int -- number of filters
    :param dropout_rate: float -- dropout rate
    :param weight_decay: float -- weight decay
    :param nb_layers:int --numbers of layers in a dense block
    :param filter_size_ori: int -- filter size of first conv1d
    :param dropout_dense: float---drop out rate of dense

    :returns: keras model with nb_layers of conv_factory appended
    :rtype: keras model

    """
    
    
    
    
    # first input of 33 seq #
    main_input = Input(shape=img_dim1)
    #import tensorflow as tf
    #main_input=encoder1(main_input)

    # Initial convolution
    x1 = Conv1D(nb_filter, filter_size_ori,
                      kernel_initializer = init_form,
                      activation='relu',
                      padding='same',
                      use_bias=False,
                      kernel_regularizer=l2(weight_decay))(main_input)#main_input
    x11 = x1
    x1 = denseblock(x1, init_form, nb_layers, nb_filter, growth_rate,filter_size_block1,
                                  dropout_rate=dropout_rate,
                                  weight_decay=weight_decay)
    

    #x1=encoder1(x1)
    x11 = x1
    #x1=non_local_attention(x1)
    #x11=encoder1(x11)

    x11 = transitionh(x11, init_form, nb_filter, dropout_rate=dropout_rate,
                       weight_decay=weight_decay)
    x1 = transition(x1, init_form, nb_filter, dropout_rate=dropout_rate,
                       weight_decay=weight_decay)
    #x1=non_local_attention(x1)
    #x11=encoder1(x11)
   
    x01=concatenate([x1,x11])
    channel = x01.shape[-1]
    x1 = layers.Dense(channel*0.25)(x1)
    x11 = layers.Dense(channel*0.25)(x11)
    x1 = Activation('relu',name='seq1')(x1)
    x11 = Activation('relu',name='seq11')(x11)
    x1 = layers.Dense(channel)(x1)
    x11 = layers.Dense(channel)(x11)
    xxx1=layers.Add()([x1, x11])
    xxx1 = tf.nn.sigmoid(xxx1)
    #xxx1=encoder1(xxx1)
    xxx1 = layers.Multiply()([x01, xxx1])
    xxx1 = denseblock(xxx1, init_form, nb_layers, nb_filter, growth_rate, filter_size_block1,
                        dropout_rate=dropout_rate,
                        weight_decay=weight_decay)

    # second input of 21 seq #
    input2 = Input(shape=img_dim2)
    #input2=encoder1(input2)

    x2 = Conv1D(nb_filter, filter_size_ori,
                kernel_initializer = init_form,
                activation='relu',
                padding='same',
                use_bias=False,
                kernel_regularizer=l2(weight_decay))(input2)
    
    x2 = denseblock(x2, init_form, nb_layers, nb_filter, growth_rate, filter_size_block2,
                        dropout_rate=dropout_rate,
                        weight_decay=weight_decay)

    
    
    x22 = x2  
    #x2=non_local_attention(x2)
    #x22=encoder1(x22)
    x22 = transitionh(x22, init_form, nb_filter, dropout_rate=dropout_rate,
                        weight_decay=weight_decay)
    x2 = transition(x2, init_form, nb_filter, dropout_rate=dropout_rate,
                        weight_decay=weight_decay)
    #x2=non_local_attention(x2)
    #x22=encoder1(x22)

    x02=concatenate([x2,x22])
    channel = x02.shape[-1]
    x2 = layers.Dense(channel*0.25)(x2)
    x22 = layers.Dense(channel*0.25)(x22)
    x2 = Activation('relu',name='seq2')(x2)
    x22 = Activation('relu',name='seq22')(x22)
    x2 = layers.Dense(channel)(x2)
    x22 = layers.Dense(channel)(x22)
    xxx2=layers.Add()([x2, x22])
    xxx2 = tf.nn.sigmoid(xxx2)
    #xxx2=encoder1(xxx2)
    xxx2 = layers.Multiply()([x02, xxx2])
    xxx2 = denseblock(xxx2, init_form, nb_layers, nb_filter, growth_rate, filter_size_block2,
                        dropout_rate=dropout_rate,
                        weight_decay=weight_decay)
    #xxx2 = denseblock(xxx2, init_form, nb_layers, nb_filter, growth_rate, filter_size_block2,
                        #dropout_rate=dropout_rate,
                        #weight_decay=weight_decay)


    # The last denseblock does not have a transition

    #third input seq3 of 15 #
    input3 = Input(shape=img_dim3)
    #input3=encoder1(input3)
    x3 = Conv1D(nb_filter, filter_size_ori,
                kernel_initializer = init_form,
                activation='relu',
                padding='same',
                use_bias=False,
                kernel_regularizer=l2(weight_decay))(input3)
    x33 = x3
    
    x3 = denseblock(x3, init_form, nb_layers, nb_filter, growth_rate, filter_size_block3,
                        dropout_rate=dropout_rate,
                        weight_decay=weight_decay)

    

    x33 = x3
    #x3=non_local_attention(x3)
    #x33=encoder1(x33)
    
    x33 = transition(x33, init_form, nb_filter, dropout_rate=dropout_rate,
                       weight_decay=weight_decay)
    x3 = transition(x3, init_form, nb_filter, dropout_rate=dropout_rate,
                       weight_decay=weight_decay)
    #x3=non_local_attention(x3)
    #x33=encoder1(x33)               
    
    x03=concatenate([x3,x33])
    channel = x03.shape[-1]
    x3 = layers.Dense(channel*0.25)(x3)
    x33 = layers.Dense(channel*0.25)(x33)
    x3 = Activation('relu',name='seq13')(x3)
    x33 = Activation('relu',name='seq133')(x33)
    x3 = layers.Dense(channel)(x3)
    x33 = layers.Dense(channel)(x33)
    xxx3=layers.Add()([x3, x33])
    xxx3 = tf.nn.sigmoid(xxx3)
    #xxx3=encoder1(xxx3)
    xxx3 = layers.Multiply()([x03, xxx3])
    
    xxx3 = denseblock(xxx3, init_form, nb_layers, nb_filter, growth_rate, filter_size_block3,
                        dropout_rate=dropout_rate,
                        weight_decay=weight_decay)

      
    # contact 3 output features #
    #x16 = concatenate([xxx1,xxx2,xxx3], axis=-2, name='contact_multi_seq')
    #channel1 = x16.shape[-1]
    #x16 = layers.Dense(channel1*0.25)(x16)
    #x16 = Activation('relu',name='seq666')(x16)
    #x16 = layers.Dense(channel)(x16)
    #x16 = tf.nn.sigmoid(x16)
    #xxx=encoder1(x16)
    #xy=non_local_attention(xy)
    #xy=synthetic_attention1(xy)
    #xxx1 = cbam_block(xxx1)
    #xxx2 = cbam_block(xxx2)
    #xxx3 = cbam_block(xxx3)
    xy = concatenate([xxx1,xxx2,xxx3], axis=-2, name='contact_multi_seq')
    #xy=synthetic_attention1(xy)
    #xy=attention_mechanism(xy)
    #xy,tt =cbam_block(xy)
    #xy = denseblock(xy, init_form, nb_layers, nb_filter, growth_rate, filter_size_block1,
                        #dropout_rate=dropout_rate,
                        #weight_decay=weight_decay)
    #xy=synthetic_attention1(xy)
    #xy=synthetic_attention1(xy)
    #xxx3 = Flatten()(xxx3)
    #xy=concatenate([xy,xxx3])
    xy = Flatten()(xy)

    xxx = Dense(dense_number,
              name ='Dense_1',
              activation='relu',
              kernel_initializer = init_form,
              kernel_regularizer=l2(weight_decay),
              bias_regularizer=l2(weight_decay))(xy)

    xxx = Dropout(dropout_dense)(xxx)
    
    #softmax
    xxx = Dense(nb_classes,
              name = 'Dense_softmax',
              activation='softmax',
              kernel_initializer = init_form,
              kernel_regularizer=l2(weight_decay),
              bias_regularizer=l2(weight_decay))(xxx)
    #xxx = Flatten()(xxx)

    
    phos_model = Model(inputs=[main_input,input2,input3], outputs=[xxx], name="multi-DenseNet")
    #feauture_dense = Model(input=[main_input, input2, input3], output=[x], name="multi-DenseNet")

    return phos_model
#