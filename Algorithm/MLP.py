# 多层感知机
from mxnet import gluon
from mxnet.gluon import nn

def MLP():
    net = nn.Sequential()
    net.add(nn.Dense(200,activation='relu'),
            nn.Dense(200,activation='relu'),
            nn.Dense(10))