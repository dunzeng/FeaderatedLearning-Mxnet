import sys
path_base = "E:\PythonProjects\Mxnet_FederatedLearning"
sys.path.append(path_base)
from mxnet import ndarray as nd
import mxnet as mx
from mxnet.gluon import nn
from mxnet import gluon
import socket
import os
import pickle
import numpy as np
import copy
import Tools
import json
path_server = "E:\PythonProjects\Mxnet_FederatedLearning\Server"

class Model_Handler():
    def __init__(self,model):
        # model为定义并初始化的nn.Block类
        self.nn_model = model
    
    def train(self):
        # 调用该函数训练模型
        pass
    
    def validate_model(self):
        # 验证模型
        pass
    
    def load_data(self):
        # 加载训练数据
        pass
