from mxnet import ndarray as nd
import mxnet as mx
from mxnet.gluon import nn
from mxnet import gluon
import json

# 返回一个由json文件定义并被初始化的神经网络
def neural_network_define(ctx, json_path="network_define.json"):
    with open(json_path,'r') as f:
        data = json.load(f)
    net = nn.Sequential()
    input_shape = tuple(data['input_shape'])
    layer_info = data['layers']
    for _,value in layer_info.items():
        if value.has_key('activation'):
            act = value['activation']
        units = value['units']
        net.add(nn.Dense(units,activation=act))
    net.initialize(mx.init.Xavier(magnitude=2.24),ctx=ctx)
    init_data = nd.random.uniform(shape=input_shape,ctx=ctx)
    net(init_data)
    return net

def conv_network_define(ctx, json_path="network_define.json"):
    with open(json_path,'r') as f:
        data = json.load(f)
    net = nn.Sequential()
    input_shape = tuple(data['input_shape'])
    layer_info = data['layers']
    # Conv2D MaxPool2D Dense
    Conv2D = "Conv2D"
    MaxPool2D = "MaxPool2D"
    Dense = "Dense"
    for key,value in layer_info.items():
        if Conv2D in key:
            pass
        elif MaxPool2D in key:
            pass
        elif Dense in key:
            pass
        else:
            raise("json Define error")
