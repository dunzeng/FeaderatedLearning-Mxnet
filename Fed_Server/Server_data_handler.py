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
from Tools.log import log
import json
 
class Server_data_handler():
    """
    模型管理类
    管理服务器端内部参数处理
    Algorithm: 模型选择
               梯度处理
    """
    def __init__(self, model, input_shape, learning_rate, init_model_path="", random_initial_model=True):
        # model: MXnet中nn.Block类或其派生类
        # data_shape: 模型输入数据形状
        # learning_rate: Server端接收梯度时的更新学习率
        # random_inital_model: 是否随机生辰初始模型
        # init_model_dir: 随机初始化模型保存路径

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
        self.model_path = init_model_path

        if random_initial_model == True:
            self.__init_model()
        else:
            try:
                self.__net.load_parameters(init_model_path,ctx=self.__ctx)
            except:
                raise ValueError("Invalid init_model_path")   
         
        # log 类
        self.log = log(path_base + "\\Fed_Server\\log")
        
    def __get_deafault_valData(self):
        mnist = mx.test_utils.get_mnist()
        val_data = {"test_data":mnist['test_data'],"test_label":mnist['test_label']}
        return val_data

    def __init_model(self):
        # 初始化用户自定义的模型
        #self.input_shape,self.__net = self.custom_model()
        self.__net.initialize(mx.init.Xavier(magnitude=2.24),ctx=self.__ctx)
        # Mxnet：神经网络在第一次前向传播时初始化
        # 因此初始化神经网络时需要做一次随机前向传播
        self.__net(nd.random.uniform(shape=self.input_shape,ctx=self.__ctx[0]))
        # 保存初始化模型 Server可发送至Client训练
        #self.__net.save_parameters(save_path)
        print("-验证Server端初始模型性能-")
        self.validate_current_model(self.__get_deafault_valData())

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

    def __update_gradient(self,gradient_info):
        # 由Client回传的梯度信息 更新Server模型
        idx = 0
        gradient_w = gradient_info['weight']
        gradient_b = gradient_info['bias']
        update_flag = False
        lr = self.learning_rate
        for layer in self.__net:
            try:
                layer.weight.data()[:] = layer.weight.data()[:] - (lr*gradient_w[idx]).as_in_context(layer.weight.data().context)
                layer.bias.data()[:] = layer.bias.data()[:] - (lr*gradient_b[idx]).as_in_context(layer.bias.data().context)
            except:
                continue
            idx += 1

            if update_flag is not True:
                update_flag = True
        
        if update_flag:
            print("-gradient successfully updated-")
        else:
            print("-oops! gradient failure-")
    
    def current_model_accepted(self,save_dir):
        try:
            print("Server模型覆盖更新",save_dir)
            self.__net.save_parameters(save_dir)
        except:
            raise ValueError("Invalid path %s"&save_dir)

    def save_current_model2file(self,save_dir):
        try:
            print("模型保存",save_dir)
            self.__net.save_parameters(save_dir)
        except:
            raise ValueError("Invalid path %s"&save_dir)
        
    def process_data_from_client(self, client_data, mode):
        # mode: replace模型替换 gradient梯度更新 defined自定义
        print("处理Client回传数据 mode: ",mode)
        if mode=='replace':
            # replace 模式下直接将传回的模型作为当前模型
            self.__net = client_data
        elif mode=='gradient':
            # 3.22 Client回传梯度
            self.__update_gradient(client_data)
        elif mode=='defined':
            # 自定义算法
            self.defined_data_method(client_data)
        else:
            raise ValueError("Invalid mode %s. Options are replace, gradient and defined"&mode)

    def defined_data_method(self,client_data):
        # 自定义算法使用处理Client数据
        pass
    
    # debug
    def get_model(self):
        return copy.deepcopy(self.__net)
    
    
