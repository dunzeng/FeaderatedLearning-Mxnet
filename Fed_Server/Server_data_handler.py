import sys
path_base = "E:\\PythonProjects\\Mxnet_FederatedLearning"
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
 
class Server_data_handler():
    """
    模型管理类
    管理服务器端内部参数处理
    Algorithm: 模型选择
               梯度处理
    """
    def __init__(self,model):
        # model为MXnet中nn.Block类或其派生类
        #初始化系统参数
        with open(path_base+"\\Fed_Server\\param_config.json",'r') as f:
            json_data = json.load(f)
        self.learning_rate = json_data['learning_rate']
        #self.init_model_path = json_data['init_model_path'] 
        self.update_model_path = json_data['updata_model_path']

        # 初始化模型
        self.__net = model
        self.input_shape = None
        self.__ctx = Tools.utils.try_all_gpus()
        #self.init_model(save_path=self.update_model_path)
    
    def init_model(self,save_path=""):
        # 初始化用户自定义的模型
        #self.input_shape,self.__net = self.custom_model()
        self.__net.initialize(mx.init.Xavier(magnitude=2.24),ctx=self.__ctx)
        # Mxnet特性：神经网络在第一次前向传播时初始化
        # 因此初始化神经网络时需要做一次随机前向传播
        self.__net(nd.random.uniform(shape=self.input_shape,ctx=self.__ctx[0]))
        # 保存初始化模型 Server可发送至Client训练
        self.__net.save_parameters(save_path)
    
    # 评估当前模型准确率
    def validate_current_model(self):
        mnist = mx.test_utils.get_mnist()
        val_data = mx.io.NDArrayIter(mnist['test_data'],mnist['test_label'],batch_size=100)
        for batch in val_data:
            data = gluon.utils.split_and_load(batch.data[0],ctx_list=self.__ctx,batch_axis=0)
            label = gluon.utils.split_and_load(batch.label[0],ctx_list=self.__ctx,batch_axis=0)
            outputs = []
            metric = mx.metric.Accuracy()
            for x in data:
                outputs.append(self.__net(x))
            metric.update(label,outputs)
        print('验证集准确率 validation acc:%s=%f'%metric.get())

    def updata_model(self, model):
        self.__net = model

    def update_gradient(self,gradient_info=None,traverse_list=[]):
        # 更新Server梯度信息
        cur_dep = 0
        gradient_w = gradient_info[0]
        gradient_b = gradient_info[1]
        for layer in self.__net:
            try:
                #朴素算法下 梯度信息为梯度列表 顺序遍历更新
                layer.weight.set_data(layer.weight.data()[:] - self.learning_rate*gradient_w[cur_dep])
                layer.bias.set_data(layer.bias.data()[:] - self.learning_rate*gradient_b[cur_dep])
                cur_dep += 1
            except:
                continue
        print('weight updated!')
    
    def current_model_accepted(self,save_dir=''):
        print("Server模型覆盖更新",self.update_model_path)
        self.__net.save_parameters(self.update_model_path)