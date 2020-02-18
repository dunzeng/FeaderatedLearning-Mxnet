import sys
path_base = "E:\PythonProjects\Mxnet_FederatedLearning"
path_server = "E:\PythonProjects\Mxnet_FederatedLearning\Server"
path_client = "E:\PythonProjects\Mxnet_FederatedLearning\Client"
sys.path.append(path_base)
sys.path.append(path_server)
sys.path.append(path_client)

import socket
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd as ag
from mxnet import ndarray as nd
import copy
import pickle
from utils import recv_class
from utils import send_class
from utils import max_parse_gradient_matrix
from utils import direct_gradient
from utils import max_parse_gradient_matrix_list
from utils import network_layers_filter
import json
from data_handler import data_handler

# 参与者类
# 维护与服务器端的通信和执行联邦学习协议
# 维护本地模型，包括参数更新、模型训练
# 
class Client:
    def __init__(self):
        # 网络通信类
        self.sock = socket.socket()
        with open("server_config.json",'r') as f:
            json_data = json.load(f)
        self.server_addr = (json_data['server_ip'],json_data['server_port'])

        # 模型处理类
        self.data_handler = data_handler()
        # Selective SSGD
        # initalized by the Server
        self.SSGD_activated = False
        self.theta = None
        self.learning_rate = None
        self.Tao_ = None
    
    # 从Json文件中初始化部分变量
    def __initialize_from_json(self):
        with open("server_config.json",'r') as f:
            json_data = json.load(f)
        self.server_addr = (json_data['server_ip'],json_data['server_port'])

        
    #创建socket连接，发送控制码，socket连接保存
    def __send_code(self,msg_code=""):
        self.sock = socket.socket()
        self.sock.connect(self.server_addr)
        self.sock.send(msg_code.encode('utf-8'))

    #从socket中获得控制码信息
    def __recv_code(self):
        msg_code = self.sock.recv(1024).decode()
        return msg_code
    
    def __init_system_meta_param(self):
        message = '1001'
        self.__send_code(message)
        param_list = recv_class(self.sock)
        print(param_list)
        self.SSGD_activated = param_list[0]
        if self.SSGD_activated is True:
            self.theta = param_list[1]
            self.learning_rate = param_list[2]
            self.Tao_ = param_list[3]

    def __check_model_type(self):
        message = '1003'
        self.__send_code(message)
        return int(self.__recv_code())
    
    def __upload_gradient(self, gradient_info):
        message = '1004'
        self.__send_code(message)
        send_class(self.sock,gradient_info)
    
    """
    请求服务器端模型用于本地训练
    """
    def __ask_for_model(self):
        if self.SSGD_activated is True:
            model_name = "\\recv_selected_model.params"
        else:
            model_name = "\\recv_model.params"
        print(model_name)
        message = '1002'
        self.__send_code(message)
        # 接收模型
        # 下载模型并写入文件
        file_size = int(self.sock.recv(1024).decode())
        has_recv = 0
        f = open(path_client+model_name,'wb')
        while has_recv != file_size:
            data = self.sock.recv(1024)
            f.write(data)
            has_recv += len(data)
        f.close()
        self.sock.close()

    # Client端流程 
    # 初始化参数->请求模型->加载模型->训练->梯度回传
    def process(self):
        self.__init_system_meta_param()
        self.__ask_for_model()
        self.data_handler.load_model()
        gradient_ = self.data_handler.local_train(50,10,0.02)
        self.__upload_gradient(gradient_)

if __name__ == "__main__":
    client = Client()
    client.process()
    
    """
    def process(self,round):
        self.__init_system_meta_param()
        self.__ask_for_model()
        flag = self.__check_model_type()
        self.reset_model(flag)
        self.__load_model()
        gradient_dict = self.__local_train(batch_size=100,epoch=10,learning_rate=0.01,round=round)
        self.__upload_gradient(gradient_dict)
    """

    """
    Private Method
    """
    #随机初始化生成模型
    # *
    """"
    def __random_init_local_model(self,generate_dir = None):
        net = nn.Sequential()
        net.add(nn.Dense(128,activation='relu'))
        net.add(nn.Dense(64,activation='relu'))
        net.add(nn.Dense(10))
        net.initialize(mx.init.Xavier(magnitude=2.24),ctx=self.ctx)
        sample = nd.random.uniform(shape=(1,28,28),ctx=mx.gpu())
        net(sample)
        net.save_parameters(self.init_model_dir)
    """

    """
    # *
    def __reset_ctx(self,ctx=[]):
        self.ctx = ctx
    """

    """
    # *
    def __data_loader(self,idx=0):
        path = "E:\PythonProjects\Mxnet_FederatedLearning\Client\local_data"
        data_file = "\\train_data_600_"+str(idx)+".npy"
        label_file = "\\train_label_600_"+str(idx)+".npy"
        data = np.load(path+data_file)
        label = np.load(path+label_file)
        return data,label
    """

    """
    # *
    def __selected_gradient(self,select_flags="",gradient_info=None):
        for key,value in gradient_info.items():
            max_parse_gradient_matrix(self.theta, value)
            gradient_info[key] = value
        return gradient_info
    """

    """
    def __load_model(self,model_dir='recv_model.params'):
        self.__net.load_parameters('recv_model.params',ctx=self.ctx)
    """

    # SSGD
    # 加载选择权值矩阵
    # 置零选择模型非零参数中对应的参数
    # 两矩阵相加
    """
    def __load_selected_model(self,model_dir='recv_selected_model.params'):
        tmp_net = self.__reset_network()
        tmp_net.load_parameters(path_client+"\\"+model_dir)
        params,depth = network_layers_filter(network=tmp_net)
        for dep in range(depth):
            data_np = params["layer"+str(dep)+"_weight"].asnumpy()
            layer_np = self.__net[dep].weight.data()[:].asnumpy()
            layer_np[data_np!=0] = 0
            layer_np += data_np
            layer_nd = nd.array(layer_np)
            self.__net[dep].weight.data()[:] = layer_nd
    """

    """
    def __reset_network(self):
        net = nn.Sequential()
        net.add(nn.Dense(128,activation='relu'))
        net.add(nn.Dense(64,activation='relu'))
        net.add(nn.Dense(10))
        net.initialize(mx.init.Xavier(magnitude=2.24),ctx=self.ctx)
        net(nd.random.uniform(shape=(1,28,28),ctx=self.ctx[0]))
        return net
    """
    
    """
    def __gradient_dict2list(self,gradient_dict):
        grad_list = []
        for _,value in gradient_dict.items():
            grad_list.append(value)
        return grad_list
    """

    """
    # Public Method   
    # for testing and debugging 
    def test_Run(self):
        for idx in range(10):
            print("round",idx)
            self.process(idx)
    """
    

    # SSGD算法参与者端流程 
    # 初始化参数->请求模型->检查模型结构->加载模型->训练->选择梯度回传
    """
    def SSGD_process(self):
        self.__init_system_meta_param()
        self.__ask_for_model()
        flag = self.__check_model_type()
        self.reset_model(flag)
        self.__load_model(model_dir=self.init_model_dir)
        self.__load_selected_model()
        grad_dict = self.__local_train()
        grad_list = self.__gradient_dict2list(grad_dict)
        grad_list_np = []
        # 全局参数筛选
        #grad_list_np = max_parse_gradient_matrix_list(self.theta,grad_list)
        # 单层参数筛选
        for grad_nd in grad_list:
            grad_list_np.append(max_parse_gradient_matrix(self.theta,grad_nd))
        grad_list_nd = []
        for grad_np in grad_list_np:
            grad_list_nd.append(nd.array(grad_np))
        self.__upload_gradient(grad_list_nd)
    """
    #batch_size设置导致传递时错误
    """
    def __local_train(self,batch_size=50,epoch=10,learning_rate=0.01,round=0):
        data,label = self.__data_loader(round)
        train_data = mx.io.NDArrayIter(data,label,batch_size,shuffle=True)
        #train
        net_origin = copy.deepcopy(self.__net)
        metric = mx.metric.Accuracy()
        loss = gluon.loss.SoftmaxCrossEntropyLoss()
        trainer = gluon.Trainer(self.__net.collect_params(),'sgd',{'learning_rate':learning_rate})
        for i in range(epoch):
            train_data.reset()
            for batch in train_data:
                data = gluon.utils.split_and_load(batch.data[0],ctx_list=self.ctx,batch_axis=0)
                label = gluon.utils.split_and_load(batch.label[0], ctx_list=self.ctx, batch_axis=0)
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
        #返回梯度字典
        return direct_gradient(net_origin,self.__net,learning_rate)
    """
    """
    # 重置模型self.__net 用于二次加载模型
    def reset_model(self,model_flag):
        #检查self.__net状态 for test_run()
        if self.model_set_flag==True:
            return
        else:
            self.model_set_flag=True
        #MLP
        if model_flag == 0:
            self.__net.add(
                #nn.Dense(784),
                nn.Dense(128,activation='relu'),
                nn.Dense(64,activation='relu'),
                nn.Dense(10)
                )
        #LeNet
        elif model_flag == 1:
            self.__net.add(nn.Conv2D(channels=6,kernel_size=5,activation='relu'),
                nn.MaxPool2D(pool_size=(2,2),strides=(2,2)),
                nn.Conv2D(channels=16,kernel_size=(5,5),strides=(1,1),padding=(0,0),activation='relu'),
                nn.Dense(units=120,activation='relu'),
                nn.Dense(units=84,activation='relu'),
                nn.Dense(units=10)
                )
        else:
            raise('模型错误')
    """
"""
if __name__ == '__main__':
    ip = ""
    #join = Participant(server_ip=ip)
    join = Client()
    #join.SSGD_process()
    #join.process(round=7)
    join.test_Run()
"""