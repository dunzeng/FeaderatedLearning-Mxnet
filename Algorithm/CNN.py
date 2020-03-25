# RNN
from mxnet import gluon
from mxnet.gluon import nn

#from LeNet import LeNet_
#from AlexNet import AlexNet_
# VGG暂时未添加
#from VGG import vgg_

# LeNet
def LeNet_(activation='relu'):
    # 获取一个结构定义完整的LeNet卷积神经网络
    # 激活函数可自选 默认为relu
    net = nn.Sequential()
    net.add(nn.Conv2D(channels=6, kernel_size=(5,5), activation=activation),
            nn.MaxPool2D(pool_size=(2,2), strides=(2,2)),
            nn.Conv2D(channels=16, kernel_size=(5,5), activation=activation),
            nn.MaxPool2D(pool_size=(2,2), strides=(2,2)),
            # Dense会默认将(批量大小， 通道， 高， 宽)形状的输入转换成
            # (批量大小， 通道 * 高 * 宽)形状的输入
            nn.Dense(120, activation=activation),
            nn.Dense(84, activation=activation),
            nn.Dense(10))
    return net

#AlexNet
def AlexNet_():
    # 获取一个结构定义完整的AlexNet卷积神经网络
    net = nn.Sequential()
    # 使用较大的11 x 11窗口来捕获物体。 同时使用步幅4来较大幅度减小输出高和宽。这里使用的输出
    # 通道数比LeNet的也要大很多
    net.add(nn.Conv2D(96, kernel_size=11, strides=4, activation='relu'),
            nn.MaxPool2D(pool_size=3, strides=2),
            # 减小卷积窗口，使用填充为2来使得输入与输出高和宽一致，且增大输出通道数
            nn.Conv2D(256, kernel_size=5, padding=2, activation='relu'),
            nn.MaxPool2D(pool_size=3, strides=2),
            # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
            # 前两个卷积层后不使用池化层来减小输入的高和宽
            nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
            nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
            nn.Conv2D(256, kernel_size=3, padding=1, activation='relu'),
            nn.MaxPool2D(pool_size=3, strides=2),
            # 这里全连接层的输出个数比LeNet中的大数倍。 使用丢弃层来缓和过拟合
            nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
            nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
            # 输出层 对应10分类
            nn.Dense(10))
    return net

#VGG
conv_arch = ((1,64), (1,128), (2,256), (2,512), (2,512))
def vgg_block(num_convs, num_channels):
    blk = nn.Sequential()
    for _ in range(num_convs): 
        blk.add(nn.Conv2D(num_channels, kernel_size=3, padding=1, activation='relu'))
    blk.add(nn.MaxPool2D(pool_size=2, strides=2))
    return blk

def vgg_(conv_arch):
    net = nn.Sequential()
    # 卷积层部分
    for (num_convs, num_channels) in conv_arch:
        net.add(vgg_block(num_convs, num_channels))
    # 全连接层
    net.add(nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
            nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
            nn.Dense(10))
    return net


def CNN_Model(mode=''):
    if mode == 'LeNet':
        return LeNet_()
    elif mode == 'AlexNet':
        return AlexNet_()
    else:
        raise ValueError("Invalid mode %s. Options are LeNet, AlexNet and VGG"&mode)

"""
class CNN_Model(nn.Block):
    # 卷积神经网络库
    def __init__(self, mode, **kawrgs):
        super(CNN_Model, self).__init__(**kawrgs)
        with self.name_scope():
            if mode == 'LeNet':
                self.cnn = LeNet_()
            elif mode == 'AlexNet':
                self.cnn = AlexNet_()
            else:
                 raise ValueError("Invalid mode %s. Options are LeNet, AlexNet and VGG"&mode)
    
    def get_iterable_model(self):
        return self.cnn
    
    def forward(self,x):
        return self.cnn(x)
"""
