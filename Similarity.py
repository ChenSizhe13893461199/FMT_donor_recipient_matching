# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:54:00 2024

@author: CHEN Sizhe
"""
#Similarity Calculation

import pandas as pd
import numpy as np
from scipy.special import kl_div
# 读取CSV文件
data = pd.read_csv('bacteria.csv', index_col=0)

# 将NA替换为0
data.fillna(0, inplace=True)

from numpy import dot
from numpy.linalg import norm

def cosine_similarity(A, B):
    return dot(A, B) / (norm(A) * norm(B))

def jaccard_similarity(A, B):
    A_nonzero = A > 0
    B_nonzero = B > 0
    intersection = np.sum(A_nonzero & B_nonzero)
    union = np.sum(A_nonzero | B_nonzero)
    return intersection / union if union != 0 else 0

import numpy as np

def bray_curtis_similarity(a, b):
    """
    计算两个样本之间的布雷依-柯提斯相似度。
    
    参数:
    a (array-like): 第一个样本的丰度向量。
    b (array-like): 第二个样本的丰度向量。
    
    返回:
    float: 布雷依-柯提斯相似度值，范围在 [0, 1] 之间。
    """
    a = np.array(a)
    b = np.array(b)
    
    # 计算丰度的绝对值之和
    sum_ab = np.sum(np.abs(a) + np.abs(b))
    # 计算丰度差的绝对值之和
    sum_diff = np.sum(np.abs(a - b))
    
    # 计算布雷依-柯提斯相似度
    similarity = 1 - (sum_diff / sum_ab)
    
    return similarity

from scipy.stats import entropy
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import canberra
# 提取两个样本的读取数#similarity calculation
i=0
a=[]
b=[]
c=[]
d=[]
for t in range(0,29):
 sample1 = data.iloc[i].values
 sample2 = data.iloc[i+1].values
 sample3 = data.iloc[i+2].values
 i=i+1
 a.append(canberra(sample1,sample2))
 b.append(canberra(sample1,sample3))
 c.append(canberra(sample2,sample3))
 d.append(canberra(sample2,sample3)-canberra(sample1,sample2))








# 提取两个样本的读取数#similarity calculation
i=0
a=[]
b=[]
c=[]
d=[]
for t in range(0,29):
 sample1 = data.iloc[i].values
 sample2 = data.iloc[i+1].values
 sample3 = data.iloc[i+2].values
 i=i+1
 a.append(cosine_similarity(sample1, sample2))
 b.append(cosine_similarity(sample1, sample3))
 c.append(cosine_similarity(sample2, sample3))
 d.append(cosine_similarity(sample2, sample3)-cosine_similarity(sample1, sample3))

# 提取两个样本的读取数#similarity calculation
i=0
a=[]
b=[]
c=[]
d=[]
for t in range(0,29):
 sample1 = data.iloc[i].values
 sample2 = data.iloc[i+1].values
 sample3 = data.iloc[i+2].values
 i=i+1
 a.append(jaccard_similarity(sample1, sample2))
 b.append(jaccard_similarity(sample1, sample3))
 c.append(jaccard_similarity(sample2, sample3))
 d.append(jaccard_similarity(sample2, sample3)-jaccard_similarity(sample1, sample3))
 
 # 提取两个样本的读取数#similarity calculation
i=0
a=[]
b=[]
c=[]
d=[]
for t in range(0,29):
 sample1 = data.iloc[i].values
 sample2 = data.iloc[i+1].values
 sample3 = data.iloc[i+2].values
 i=i+1
 a.append(bray_curtis_similarity(sample1, sample2))
 b.append(bray_curtis_similarity(sample1, sample3))
 c.append(bray_curtis_similarity(sample2, sample3))
 d.append(bray_curtis_similarity(sample2, sample3)-bray_curtis_similarity(sample1, sample3))
 
 
## 将列表转换为 NumPy 数组
#d_np = np.array(d)

# 计算大于 0 的数值个数
#count_greater_than_zero = np.sum(d_np > 0)
cosine_sim = cosine_similarity(sample1, sample2)
print(f'Cosine Similarity: {cosine_sim}')

cosine_sim = cosine_similarity(sample1, sample3)
print(f'Cosine Similarity: {cosine_sim}')

cosine_sim = cosine_similarity(sample2, sample3)
print(f'Cosine Similarity: {cosine_sim}')


jaccard_sim = jaccard_similarity(sample1, sample2)
print(f'Jaccard Similarity: {jaccard_sim}')

jaccard_sim = jaccard_similarity(sample1, sample3)
print(f'Jaccard Similarity: {jaccard_sim}')

jaccard_sim = jaccard_similarity(sample2, sample3)
print(f'Jaccard Similarity: {jaccard_sim}')


similarity = bray_curtis_similarity(sample1, sample2)
print(f"布雷依-柯提斯相似度: {similarity:.4f}")

similarity = bray_curtis_similarity(sample1, sample3)
print(f"布雷依-柯提斯相似度: {similarity:.4f}")

similarity = bray_curtis_similarity(sample2, sample3)
print(f"布雷依-柯提斯相似度: {similarity:.4f}")
