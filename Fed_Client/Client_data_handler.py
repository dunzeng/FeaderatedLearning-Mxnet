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
import json 

class Client_data_handler:
    # 架构第二层
    # 该类直接作为网络类Client的成员
    # 负责数据处理 
    def __init__(self, model):
        # model为用户自定义网络模型类 其类型应为MXnet中nn.block类
        # 模型初始化
        self.__ctx = Tools.utils.try_all_gpus()
        self.__model = model
        self.input_shape = None
        # 初始化存储路径
        with open(path_base+"\\Fed_Client\\data_handler_config.json",'r') as f:
            json_data = json.load(f)
        self.model_save_path = ""
        self.model_load_path = ""  
        self.init_model_path = ""
        self.local_data_file = (json_data['local_data_path'],json_data['local_label_path'])

    def init_model(self,save_path=""):
        # 随机初始化用户自定义的模型
        #self.input_shape,self.__net = self.custom_model()
        self.__model.initialize(mx.init.Xavier(magnitude=2.24),ctx=self.__ctx)
        self.__model(nd.random.uniform(shape=self.input_shape,ctx=self.__ctx[0]))
    
    def load_model(self,model_path):
        self.__model.load_parameters(model_path,ctx=self.__ctx)

    def train_data_loader(self):
        # 用户需重写该函数用于读取模型
        # 返回元组(data,label)
        # 用于调用mx.io.NDArrayIter(data,label,batch_size)
        data = np.load(self.local_data_file[0])
        label = np.load(self.local_data_file[1])
        return data,label

    def local_train(self,batch_size,epoch,learning_rate):
        # 可由用户重写
        # 利用本地数据训练模型
        # 返回神经网络梯度信息
        # 保留已训练好的模型
        data,label = self.train_data_loader()
        train_data = mx.io.NDArrayIter(data,label,batch_size=batch_size,shuffle=True)
        metric = mx.metric.Accuracy()
        loss = gluon.loss.SoftmaxCrossEntropyLoss()
        trainer = gluon.Trainer(self.__model.collect_params(),'sgd',{'learning_rate':learning_rate})
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
                trainer.step(batch.data[0].shape[0])
            name, acc = metric.get()
            metric.reset()
            print('training acc at epoch %d/%d: %s=%f'%(i+1,epoch, name, acc))
    
    def get_model(self):
        return copy.deepcopy(self.__model)