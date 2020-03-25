import sys
path_base = "E:\\PythonProjects\\Mxnet_FederatedLearning"
sys.path.append(path_base)
import socket
import copy
import pickle
import json
from Fed_Client.Client_data_handler import Client_data_handler
from Tools import utils

# 参与者类
# 维护与服务器端的通信和执行联邦学习协议
# 维护本地模型，包括参数更新、模型训练
class Client:
    def __init__(self,data_handler):
        # 网络通信类
        self.sock = socket.socket()
        with open(path_base+"\\Fed_Client\\client_config.json",'r') as f:
            json_data = json.load(f)
        self.server_addr = (json_data['server_ip'],json_data['server_port'])
        self.recv_model_savepath = json_data['default_path']
        # 模型处理类
        self.data_handler = data_handler
        # 训练模式 从Server端同步获取
        self.train_mode = ""
        self.batch_size = None
        self.epoch = None
    
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

    def __upload_information(self, information):
        # 用户可自定义内容
        # 向客户端传送信息
        message = '1004'
        print("正向Server端发送信息 ",message)
        self.__send_code(message)
        utils.send_class(self.sock, information)
        self.sock.close()

    def __ask_for_model(self,save_path=""):
        # 向服务端请求模型
        # 用于本地训练
        message = '1002'
        print("正向Server端请求模型 ",message)
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
        print("模型下载结束 ")
    
    def __param_sync(self):
        message = '1003'
        self.__send_code(message)
        # 同步系统参数
        server_info = utils.recv_class(self.sock)
        print("同步参数train_mode: ",server_info[0])
        self.train_mode=server_info[0]

    def process(self,mode=''):
        # Client端流程 
        # 初始化参数->请求模型->加载模型->训练->梯度回传
        # 考虑不同算法 朴素Fed,FedAvg回传信息时的处理
        print("******Phase 1******")
        self.__ask_for_model(self.recv_model_savepath)
        print("******Phase 2******")
        self.__param_sync()
        print("******Phase 3******")
        self.data_handler.load_model(self.recv_model_savepath)
        print("******Phase 4******")
        self.data_handler.local_train(batch_size=100,learning_rate=0.02,epoch=10)
        print("******Phase 5******")
        # 根据训练模式不同 选择回传梯度或者模型
        if self.train_mode=='gradient':
            grad_info = self.data_handler.get_gradient()
            self.__upload_information(grad_info)
        elif self.train_mode=='replace':
            model_info = self.data_handler.get_model()
            self.__upload_information(model_info)
        elif self.train_mode=='defined':
            defined_info = None
            self.__upload_information(defined_info)
        else:
            raise ValueError("Invalid mode %s. Options are replace, gradient and defined"&mode)