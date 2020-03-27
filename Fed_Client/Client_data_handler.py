import sys
path_base = "E:\\PythonProjects\\Mxnet_FederatedLearning"
sys.path.append(path_base)

import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd as ag
from mxnet import ndarray as nd
import numpy as np
import copy
import Tools
from Tools.log import log
import json 
import time

class Client_data_handler:
    # 架构第二层
    # 该类直接作为网络类Client的成员
    # 负责数据处理 
    def __init__(self, model, input_shape, train_data_path=None):
        # model为用户自定义网络模型类 其类型应为MXnet中nn.block类
        # 模型初始化
        self.__model = model
        self.input_shape = input_shape
        self.__ctx = Tools.utils.try_all_gpus()
        self.__random_init_model()
        
        # 初始化存储路径
        with open(path_base+"\\Fed_Client\\data_handler_config.json",'r') as f:
            json_data = json.load(f)
        self.local_data_file = (json_data['local_data_path'],json_data['local_label_path'])

        # 本地梯度维护
        self.local_gradient = {"weight":[],"bias":[]}
        self.__init_gradient_list()
        
        # 本地训练数据路径
        self.train_data_path = train_data_path

        # log类
        self.log = log(path_base+"\\Fed_Server\\log")

        # 初始化log日志载入

    def __init_gradient_list(self):
        for layer in self.__model:
            try:
                shape_w = layer.weight.data().shape
                shape_b = layer.bias.data().shape
            except:
                continue
            self.local_gradient['weight'].append(nd.zeros(shape=shape_w,ctx=self.__ctx[0]))
            self.local_gradient['bias'].append(nd.zeros(shape=shape_b,ctx=self.__ctx[0]))

        #print("梯度信息shape列表：")
        #for grad in self.local_gradient:
        #    print(grad.shape)

    def __random_init_model(self):
        # 随机初始化用户自定义的模型
        #self.input_shape,self.__net = self.custom_model()
        self.__model.initialize(mx.init.Xavier(magnitude=2.24),ctx=self.__ctx)
        self.__model(nd.random.uniform(shape=self.input_shape,ctx=self.__ctx[0]))
    
    def load_model(self,model_path):
        print("加载模型 ",model_path)
        self.__model.load_parameters(model_path,ctx=self.__ctx)

    def train_data_loader(self):
        # 用户需重写该函数用于读取模型
        # 返回元组(data,label)
        # 用于调用mx.io.NDArrayIter(data,label,batch_size)
        if self.train_data_path!=None:
            data = np.load(self.train_data_path['data'])
            label = np.load(self.train_data_path['label'])
        else:
            data = np.load(self.local_data_file[0])
            label = np.load(self.local_data_file[1])
        return data,label

    def __gradient_collect(self,batch_size):
        # local_train中调用 从model中收集梯度信息
        idx = 0
        for layer in self.__model:
            try:
                grad_w = layer.weight.data().grad
                grad_b = layer.bias.data().grad
            except:
                continue
            # as_in_context() 使tensor处于同一u下运算
            self.local_gradient['weight'][idx] += grad_w.as_in_context(self.local_gradient['weight'][idx])/batch_size
            self.local_gradient['bias'][idx] += grad_b.as_in_context(self.local_gradient['bias'][idx])/batch_size
            idx+=1

    def local_train(self,batch_size,learning_rate=0.02,train_data=None,epoch=10,train_mode='gradient'):
        # 可由用户重写
        # 利用本地数据训练模型
        # 返回神经网络梯度信息
        # 保留已训练好的模型
        print("本地训练 batch_size:%d - learning_rate:%f"%(batch_size,learning_rate))
        #data,label = train_data['data'],train_data['label']
        data,label = self.train_data_loader()
        train_data = mx.io.NDArrayIter(data,label,batch_size=batch_size,shuffle=True)
        loss = gluon.loss.SoftmaxCrossEntropyLoss()
        trainer = gluon.Trainer(self.__model.collect_params(),'sgd',{'learning_rate':learning_rate})
        metric = mx.metric.Accuracy()
        for i in range(epoch):
            train_data.reset()
            for batch in train_data:
                data = gluon.utils.split_and_load(batch.data[0],ctx_list=self.__ctx,batch_axis=0)
                label = gluon.utils.split_and_load(batch.label[0], ctx_list=self.__ctx, batch_axis=0)
                outputs = []
                with ag.record():
                    for x,y in zip(data,label):
                        z = self.__model(x)
                        t_loss = loss(z,y)
                        t_loss.backward()
                        outputs.append(z)
                metric.update(label,outputs)
                # gradient mode：梯度信息采集
                if train_mode == 'gradient':
                    self.__gradient_collect(batch_size)
                trainer.step(batch_size)
            name, acc = metric.get()
            metric.reset()
            print('training acc at epoch %d/%d: %s=%f'%(i+1,epoch, name, acc))
    
    def get_model(self):
        return copy.deepcopy(self.__model)

    def get_gradient(self):
        # 上层获取梯度后 模型梯度数据清0
        gradient = copy.deepcopy(self.local_gradient)
        #载入日志文件
        self.log.new_log_file("grad"+str(int(time.time()))+".txt",gradient)
        #可优化
        self.local_gradient['weight'].clear()
        self.local_gradient['bias'].clear()
        self.__init_gradient_list()
        return gradient