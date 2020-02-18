"""
import mxnet.ndarray as nd
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd as ag
import mxnet as mx
from neural_networks import nn_mlp
import os
import sys
import tqdm
"""
import numpy as np

def max_parse(theta, matrix):
    sort_list = []
    for x in np.nditer(matrix):
        sort_list.append(abs(x))
    sort_list.sort(reverse=True)
    #print(sort_list)
    num = len(sort_list)
    for x in np.nditer(matrix,op_flags=['readwrite']):
        idx = sort_list.index(abs(x))
        #print(idx,num,idx/num)
        if idx/num >= theta :
            x[...] = 0
    #?
    #return matrix

if __name__ == "__main__":
    ls = [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
      [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]
    ls = np.array(ls)
    
    max_parse(0.5,ls)
    print(ls)
    #print(mat)