import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy

### kernel ###
def rbf_kernel(pattern1, pattern2, kernel_width):
    size1 = pattern1.size()
    size2 = pattern2.size()

    G = torch.sum(pattern1*pattern1, 1).reshape(size1[0],1)
    H = torch.sum(pattern2*pattern2, 1).reshape(size2[0],1)

    Q = torch.tile(G, (1, size2[0]))
    R = torch.tile(H.T, (size1[0], 1))

    H = Q + R - 2* (pattern1@pattern2.T)
    H = torch.exp(-H/2/(kernel_width**2))

    return H

def laplace_kernel(pattern1, pattern2, kernel_width):
    size1 = pattern1.size()
    size2 = pattern2.size()
    
    H = torch.cdist(pattern1, pattern2, p=1)
    H = torch.exp(-H/kernel_width)

    return H

def kernel_midwidth_rbf(X,Y):
    
    # ----- width of X -----
    
    n = len(X)  #样本个数#
    Xmed = X
    
#高效计算样本间的平方欧式距离#
    
    G = torch.sum(Xmed*Xmed, 1).reshape(n,1)   
    #torch:*逐个元素乘积 sum(,1)沿着第一维求和 reshape()转置#
    Q = torch.tile(G, (1, n) ) 
    #torch.tile(G, (1, n))：将 G 在第0维复制1次（不复制），在第1维复制 n 次#
    R = torch.tile(G.T, (n, 1) ) 
    #G.T(行向量 GT G转置) torch.tile复制#

    #欧氏距离矩阵#
    dists = Q + R - 2* (Xmed@Xmed.T) 
    #Xmed@Xmed内积矩阵(xiTxj)#
    
#高效计算样本间的平方欧式距离#
    
    #将距离矩阵的下三角部分（包括对角线）置零，保留上三角部分#
    dists = dists - torch.tril(dists)
    #torch.tril(dists) 会取下三角部分（包括对角线）#
    dists = dists.reshape(n**2, 1)
    #将矩阵展平为一列向量，便于后续操作#

    #计算基于中位数距离的 RBF 核宽度#
    width_x = torch.sqrt( 0.5 * torch.median(dists[dists>0]))
    #计算基于最大值距离的 RBF 核宽度#
    width_x_max = torch.sqrt( 0.5 * torch.max(dists[dists>0]))

    # ----- width of Y -----
    Ymed = Y
    G = torch.sum(Ymed*Ymed, 1).reshape(n,1)
    Q = torch.tile(G, (1, n) )
    R = torch.tile(G.T, (n, 1) )
    dists = Q + R - 2* (Ymed@Ymed.T)
    dists = dists - torch.tril(dists)
    dists = dists.reshape(n**2, 1)
    width_y = torch.sqrt( 0.5 * torch.median(dists[dists>0]))
    width_y_max = torch.sqrt( 0.5 * torch.max(dists[dists>0]))
    
    
    return width_x, width_y, width_x_max, width_y_max

def kernel_midwidth_lap(X,Y):
    
    n = len(X)
    # ----- width of X -----
    Xmed = X

    dists = torch.cdist(Xmed,Xmed,p=1)
    dists = dists - torch.tril(dists)
    dists = dists.reshape(n**2, 1)

    width_x = torch.median(dists[dists>0])   
    width_x_max = torch.max(dists[dists>0])

    # ----- width of Y -----
    Ymed = Y

    dists = torch.cdist(Ymed,Ymed,p=1)
    dists = dists - torch.tril(dists)
    dists = dists.reshape(n**2, 1)

    width_y = torch.median(dists[dists>0])
    width_y_max = torch.max(dists[dists>0])
    
    return width_x, width_y, width_x_max, width_y_max