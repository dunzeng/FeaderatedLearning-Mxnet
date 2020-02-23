import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd as ag
from mxnet import ndarray as nd
import numpy as np
import copy
from utils import network_layers_filter
from utils import direct_gradient
import json 

class Client_data_handler:
    def __init__(self,init_model_path=""):
        # 模型初始化
        self.__ctx = [mx.gpu()]
        self.__net = None
        self.input_shape = None
        self.init_model()
        # 初始化存储路径
        with open("E:\PythonProjects\Mxnet_FederatedLearning\Client\data_handler_config.json",'r') as f:
            json_data = json.load(f)
        self.model_save_path = ""
        self.model_load_path = ""
        self.init_model_path = ""
        self.local_data_file = (json_data['local_data_path'],json_data['local_label_path'])

    def custom_model(self):
        # 用户重写该函数用于生成自定义模型
        # 应返回一个结构被完全定义的nn.Sequential类/nn.Block类
        # 以及用户定义的输入数据结构shape
        # example：
        net = nn.Sequential()
        net.add(nn.Dense(128,activation='relu'))
        net.add(nn.Dense(64,activation='relu'))
        net.add(nn.Dense(10))   
        input_shape = (1,28,28)
        return input_shape,net

    def data_loader(self):
        # 用户可重写该函数用于读取模型
        # 返回元组(data,label)
        # 用于调用mx.io.NDArrayIter(data,label,batch_size)
        data = np.load(self.local_data_file[0])
        label = np.load(self.local_data_file[1])
        return data,label
    
    def init_model(self,save_path=""):
        # 初始化用户自定义的模型
        self.input_shape,self.__net = self.custom_model()
        self.__net.initialize(mx.init.Xavier(magnitude=2.24),ctx=self.__ctx)
        self.__net(nd.random.uniform(shape=self.input_shape,ctx=self.__ctx[0]))
    
    def load_model(self,model_path=""):
        # 将model_path指向的模型文件
        # 加载入self.__net
        self.__net.load_parameters(model_path)

    def local_train(self,batch_size,epoch,learning_rate):
        # 可由用户重写
        # 利用本地数据训练模型
        # 返回神经网络梯度信息
        # 保留已训练好的模型
        old_net = copy.deepcopy(self.__net)
        data,label = self.data_loader()
        train_data = mx.io.NDArrayIter(data,label,batch_size=batch_size,shuffle=True)
        metric = mx.metric.Accuracy()
        loss = gluon.loss.SoftmaxCrossEntropyLoss()
        trainer = gluon.Trainer(self.__net.collect_params(),'sgd',{'learning_rate':learning_rate})
        for i in range(epoch):
            train_data.reset()
            for batch in train_data:
                data = gluon.utils.split_and_load(batch.data[0],ctx_list=self.__ctx,batch_axis=0)
                label = gluon.utils.split_and_load(batch.label[0], ctx_list=self.__ctx, batch_axis=0)
                outputs = []
                with ag.record():
                    for x,y in zip(data,label):
                        z = self.__net(x)
                        t_loss = loss(z,y)
                        t_loss.backward()
                        outputs.append(z)
                metric.update(label,outputs)
                trainer.step(batch.data[0].shape[0])
            name, acc = metric.get()
            metric.reset()
            print('training acc at epoch %d/%d: %s=%f'%(i+1,epoch, name, acc))
        return direct_gradient(old_net,self.__net,learning_rate)
        