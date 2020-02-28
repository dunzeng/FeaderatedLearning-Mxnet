import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd as ag
from mxnet import ndarray as nd

def LeNet_(activation='sigmoid'):
    # 获取一个结构定义完整的LeNet卷积神经网络
    # 激活函数可自选 默认为sigmoid
    net = nn.Sequential()
    net.add(nn.Conv2D(channels=6, kernel_size=5, activation=activation),
            nn.MaxPool2D(pool_size=2, strides=2),
            nn.Conv2D(channels=16, kernel_size=5, activation=activation),
            nn.MaxPool2D(pool_size=2, strides=2),
            # Dense会默认将(批量大小， 通道， 高， 宽)形状的输入转换成
            # (批量大小， 通道 * 高 * 宽)形状的输入
            nn.Dense(120, activation=activation),
            nn.Dense(84, activation=activation),
            nn.Dense(10))
    return net