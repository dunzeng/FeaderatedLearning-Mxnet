from mxnet import gluon
from mxnet.gluon import nn

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