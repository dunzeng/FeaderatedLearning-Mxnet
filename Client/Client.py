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
        with open("client_config.json",'r') as f:
            json_data = json.load(f)
        self.server_addr = (json_data['server_ip'],json_data['server_port'])
        self.recv_model_savepath = json_data['default_path']
        # 模型处理类
        self.data_handler = data_handler()

        # Selective SSGD
        # initalized by the Server
        self.SSGD_activated = False
        self.theta = None
        self.learning_rate = None
        self.Tao_ = None
    
    def __initialize_from_json(self):
        # 从Json文件中初始化部分变量
        with open("server_config.json",'r') as f:
            json_data = json.load(f)
        self.server_addr = (json_data['server_ip'],json_data['server_port'])

    
    def __send_code(self,msg_code=""):
        # 创建socket连接，发送控制码
        # socket连接将被保存在self.sock中
        self.sock = socket.socket()
        self.sock.connect(self.server_addr)
        self.sock.send(msg_code.encode('utf-8'))

    
    def __recv_code(self):
        # 从socket中获得控制码信息
        # 控制码大小为4bit
        msg_code = self.sock.recv(4).decode()
        return msg_code
    
    def __upload_gradient(self, gradient_info):
        message = '1004'
        #获得socket连接
        self.__send_code(message)
        send_class(self.sock,gradient_info)
        self.sock.close()
    
    def __ask_for_model(self,save_path=""):
        # 向服务端请求模型
        # 用于本地训练
        message = '1002'
        # 获得socket连接
        self.__send_code(message)
        # 接收模型
        # 下载模型并写入文件
        file_size = int(self.sock.recv(1024).decode())
        has_recv = 0
        f = open(save_path,'wb')
        while has_recv != file_size:
            data = self.sock.recv(1024)
            f.write(data)
            has_recv += len(data)
        f.close()
        self.sock.close()

    def process(self):
        # Client端流程 
        # 初始化参数->请求模型->加载模型->训练->梯度回传
        self.__ask_for_model(self.recv_model_savepath)
        self.data_handler.load_model(self.recv_model_savepath)
        gradient_ = self.data_handler.local_train(50,10,0.02)
        self.__upload_gradient(gradient_)

if __name__ == "__main__":
    client = Client()
    client.process()
    