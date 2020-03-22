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
from Tools import utils
import json
 
class Server_data_handler():
    """
    模型管理类
    管理服务器端内部参数处理
    Algorithm: 模型选择
               梯度处理
    """
    def __init__(self, model, input_shape, learning_rate):
        # model为MXnet中nn.Block类或其派生类
        # data_shape为模型是输入数据形状
        #with open(path_base+"\\Fed_Server\\param_config.json",'r') as f:
            # 初始化系统参数
            #json_data = json.load(f)
        #self.learning_rate = json_data['learning_rate']
        #init_model_path = json_data['init_model_path'] 
        #self.update_model_path = json_data['updata_model_path']

        # 初始化模型
        self.__net = model
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.__ctx = utils.try_all_gpus()
        self.init_model()
        #self.init_model(save_path=self.update_model_path)
    
    def init_model(self,save_path=""):
        # 初始化用户自定义的模型
        #self.input_shape,self.__net = self.custom_model()
        self.__net.initialize(mx.init.Xavier(magnitude=2.24),ctx=self.__ctx)
        # Mxnet：神经网络在第一次前向传播时初始化
        # 因此初始化神经网络时需要做一次随机前向传播
        self.__net(nd.random.uniform(shape=self.input_shape,ctx=self.__ctx[0]))
        # 保存初始化模型 Server可发送至Client训练
        #self.__net.save_parameters(save_path)
    
    def validate_current_model(self,val_data_set=None):
        # 给定数据集测试模型性能
        # 评估当前模型准确率
        #val_x,val_y = val_data_set[0],val_data_set[1]
        #val_data = mx.io.NDArrayIter(val_x,val_y,batch_size=100)
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

    def update_gradient(self,gradient_info=None):
        # 由Client回传的梯度信息 更新Server模型
        idx = 0
        #gradient_w = gradient_info[0]
        #gradient_b = gradient_info[1]
        gradient_w = gradient_info
        for layer in self.__net:
            try:
                #朴素算法下 梯度信息为梯度列表 顺序遍历更新
                layer.weight.set_data(layer.weight.data()[:] - self.learning_rate*gradient_w[idx])
                #layer.bias.set_data(layer.bias.data()[:] - self.learning_rate*gradient_b[idx])
                idx += 1
            except:
                continue
        print('weight updated successfully!')
    
    def current_model_accepted(self,save_dir):
        print("Server模型覆盖更新",save_dir)
        self.__net.save_parameters(save_dir)

    def save_current_model2file(self,save_dir):
        print("模型保存",save_dir)
        self.__net.save_parameters(save_dir)

    def process_data_from_client(self, client_data, mode='replace'):
        # mode: replace模型替换 gradient梯度更新 defined自定义
        if mode=='replace':
            # replace 模式下直接将传回的模型作为当前模型
            self.__net = client_data
        elif mode=='gradient':
            # gradient模式下提取模型梯度
            #self.grab_gradient(client_data)
            #3.22 update Client回传梯度
            self.update_gradient(client_data)
        elif mode=='defined':
            self.defined_data_method(client_data)
        else:
            raise ValueError("Invalid mode %s. Options are replace, gradient and defined"&mode)

    def grab_gradient(self, client_data):
        # 计算Client端数据集整体梯度
        # lr: learning rate
        gradient_list = []
        for layer1,layer2 in self.__net,client_data:
            try:
                gradient = (layer1.weight.data()[:]-layer2.weight.data()[:])/self.learning_rate
                gradient_list.append(gradient)
            except:
                pass
    
    def defined_data_method(self,client_data):
        # 自定义算法使用处理Client数据
        pass
