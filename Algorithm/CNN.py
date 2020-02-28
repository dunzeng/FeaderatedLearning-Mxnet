# RNN
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd as ag
from mxnet import ndarray as nd
from mxnet.gluon import loss 

from LeNet import LeNet_
from AlexNet import AlexNet_
# VGG暂时未添加
from VGG import vgg_



class CNN_Model(nn.Block):
    # 卷积神经网络库
    def __init__(self, mode, **kawrgs):
        """
        mode: 选择内置卷积神经网络类型
        """
        super(CNN_Model, self).__init__(**kawrgs)
        with self.name_scope():
            if mode == 'LeNet':
                self.cnn = LeNet_()
            elif mode == 'AlexNet':
                self.cnn = AlexNet_()
            else:
                 raise ValueError("Invalid mode %s. Options are LeNet, AlexNet and VGG"&mode)
    
    def forward(self,x):
        return self.cnn(x)
