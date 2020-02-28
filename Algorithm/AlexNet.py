from mxnet import gluon
from mxnet.gluon import nn


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