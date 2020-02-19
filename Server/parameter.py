from mxnet import ndarray as nd
import mxnet as mx
from mxnet.gluon import nn
from mxnet import gluon
import socket
import os
import pickle
import numpy as np
import copy
import sys
from utils import network_layers_filter
import json
path_server = "E:\PythonProjects\Mxnet_FederatedLearning\Server"

"""
模型管理类
管理服务器端内部参数处理
"""
class Server_data_handler():
    def __init__(self,update_model_path=""):
        #初始化系统参数
        with open(path_server+"\param_config.json",'r') as f:
            json_data = json.load(f)
        self.learning_rate = json_data['learning_rate']
        self.init_model_path = json_data['init_model_path'] 
        self.update_model_path = update_model_path

        # 初始化模型
        self.__net = None
        self.input_shape = None
        self.__ctx = [mx.gpu()]
        self.init_model(save_path=self.update_model_path)

        # 模型解析
        self.params,self.depth = network_layers_filter(self.__net)
        
        """
        # Selective SGD
        # paper： Privacy-Preserving Deep Learning
        # 参数解释
        SSGD = json_data['SSGD']
        self.SSGD_activated = bool(SSGD['SSGD_act']=='True')
        if self.SSGD_activated:
            self.stat = []
            self.max_stat = 0
            self.__init_stat()
            self.theta_upload = SSGD['theta_u']
            self.theta_download = SSGD['theta_d']
            self.lambda_ = SSGD['lambda']
            self.tao = SSGD['tao']
        """
    
    def custom_model(self):
        # 用户重写该函数用于生成自定义模型
        # 应返回一个结构被完全定义的nn.Sequential类
        # 以及用户定义的输入数据shape
        # example：
        net = nn.Sequential()
        net.add(nn.Dense(128,activation='relu'))
        net.add(nn.Dense(64,activation='relu'))
        net.add(nn.Dense(10))
        input_shape = (1,28,28)
        return input_shape,net
    
    def init_model(self,save_path=""):
        # 初始化用户自定义的模型
        self.input_shape,self.__net = self.custom_model()
        self.__net.initialize(mx.init.Xavier(magnitude=2.24),ctx=self.__ctx)
        # Mxnet特性：神经网络在第一次前向传播时初始化
        # 因此初始化神经网络时需要做一次随机前向传播
        self.__net(nd.random.uniform(shape=self.input_shape,ctx=self.__ctx[0]))
        # 保存初始化模型 用户发送至Client训练
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
    
    """
    #模型初始化
    def get_LeNet(self,net_dir):
        net = nn.Sequential()
        #LeNet-5 in CIFAR-10
        net.add(nn.Conv2D(channels=6,kernel_size=5,activation='relu'),
                nn.MaxPool2D(pool_size=(2,2),strides=(2,2)),
                nn.Conv2D(channels=16,kernel_size=(5,5),strides=(1,1),padding=(0,0),activation='relu'),
                nn.Dense(units=120,activation='relu'),
                nn.Dense(units=84,activation='relu'),
                nn.Dense(units=10)
                )
        try:
            net.load_parameters(net_dir,ctx=self.ctx)
        except:
            print("初始化模型失败：",net_dir)
            raise
        return net,network_layers_filter(net)
    
    # Mnist MLP 
    # Mxnet更新 网络第一层结构由输入数据时自定义
    def __get_MLP(self,net_dir):
        net = nn.Sequential()
        #net.add(nn.Dense(784))
        net.add(nn.Dense(128,activation='relu'))
        net.add(nn.Dense(64,activation='relu'))
        net.add(nn.Dense(10))
        #net.initialize(mx.init.Xavier(magnitude=2.24),ctx=self.ctx)
        try:
            net.load_parameters(net_dir,ctx=self.ctx)
        except:
            print("初始化模型失败：",net_dir)
            raise
        params,depth = network_layers_filter(net)
        return net,params,depth
    """

    def update_gradient(self,gradient_info=None,traverse_list=[]):
        # 更新本地梯度信息
        cur_dep = 0
        gradient_w = gradient_info[0]
        gradient_b = gradient_info[1]
        while cur_dep != self.depth:
            try:
                #朴素算法下 梯度信息为梯度列表 顺序遍历更新
                self.__net[cur_dep].weight.data()[:] = self.__net[cur_dep].weight.data()[:] - self.learning_rate*gradient_w[cur_dep]
                self.__net[cur_dep].bias.data()[:] = self.__net[cur_dep].bias.data()[:] - self.learning_rate*gradient_b[cur_dep]
            except:
                break
            cur_dep += 1
        print('weight updated!')

    def current_model_accepted(self,save_dir=''):
        print("本地模型更新",self.update_model_path)
        self.__net.save_parameters(self.update_model_path)













"""

    # SSGD 
    # 根据stat分片当前模型参数
    # 将最近更新的参数取出
    # 新算法:采用网络将权值矩阵列表传出×
    # 存入模型中 将模型传出
    def get_selected_model(self, save_model_dir='selected_model.params',threshold=0):
        if self.SSGD_activated is not True:
            raise('SSGD is not ACTIVATED')
        tmp_net = copy.deepcopy(self.__net)
        for i in range(self.depth):
            print("Selecting layer %d"%(i))
            #numpy version
            data_np = tmp_net[i].weight.data().asnumpy()

            # 将最近更新的权值矩阵选出
            # 置零更新频率低的参数
            data_np[self.stat[i]<=self.max_stat-threshold] = 0

            data_nd = nd.array(data_np)
            tmp_net[i].weight.data()[:] = data_nd
        tmp_net.save_parameters(save_model_dir)

    # stat维护 stat数组记录对应权值参数被更新的次数
    def __init_stat(self):
        for i in range(self.depth):
            this_shape = self.params["layer"+str(i)+"_weight"].shape
            self.stat.append(np.zeros(shape=this_shape))
            
    # 更新stat参数 在收到回传梯度时调用
    def __update_stat_mat(self, grad_mat):
        idx = 0
        for _ ,value in grad_mat.items():
            mat_np = value.asnumpy()
            mat_np[mat_np!=0] = 1
            self.stat[idx] = self.stat[idx] + nd.array(mat_np)
            idx+=1

"""