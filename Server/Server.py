import sys
path_base = "E:\PythonProjects\Mxnet_FederatedLearning"
path_server = "E:\PythonProjects\Mxnet_FederatedLearning\Server"
path_client = "E:\PythonProjects\Mxnet_FederatedLearning\Client"
sys.path.append(path_base)
sys.path.append(path_server)
sys.path.append(path_client)

from mxnet import ndarray as nd
import mxnet as mx
from mxnet.gluon import nn
from mxnet import gluon
import socket
import os
import pickle
from parameter import Server_data_handler
import json

from utils import network_layers_filter
from utils import send_class
from utils import recv_class

"""
网络服务器：
维护网络通信，与Client交流
与后台数据处理类进行信息交换
"""

class Sever():
    def __init__(self):
        #从Json文件中读取系统配置
        with open(path_server+"\server_config.json",'r') as f:
            json_data = json.load(f)
        self.port = json_data['port']
        self.host = json_data['host']
        self.__params = Server_data_handler()
        self.sock = socket.socket()
        self.current_model_dir = path_server+"\current_model.params"
    
    """
    Private Method
    """        

    #将最新的global model发送至参与者
    def __send_model(self,connection):
        #SSGD选择参数更新
        if self.__params.SSGD_activated is True:
            SSGD_path = path_server+"\selected_model.params"
            self.__params.get_selected_model(save_model_dir=SSGD_path,threshold=0)
            file_size = self.__get_model_size(SSGD_path)
            file_dir = SSGD_path
        # 从文件中读取发送模型
        else:
            file_size = self.__get_model_size(self.current_model_dir)
            file_dir = self.current_model_dir
        print("发送模型大小：",file_size)
        connection.send(str(file_size).encode('utf-8'))
        sent_size = 0
        f = open(file_dir,'rb')
        while sent_size != file_size:
            data = f.read(1024)
            connection.send(data)
            sent_size += len(data)
        f.close()

    """
    # 发送模型确认码
    def __send_model_check(self,connection):
        message = self.model_status
        connection.send(str(message).encode('utf-8'))
    """

    # 接收参与者回传梯度
    def __recv_gradient(self,connection):
        gradient_dict = recv_class(connection)
        return gradient_dict

    # 初始化参与者参数
    def __init_paticipant(self,connection):
        if self.__params.SSGD_activated is True:
            data_list = [True,self.__params.theta_upload,self.__params.learning_rate,self.__params.tao]
            send_class(connection,data_list)
        else:
            data_list = [False]
            send_class(connection,data_list)
    
    # 创建socket连接，发送控制码，socket连接保存
    def __send_code(self,connection,msg_code=""):
        connection.send(msg_code.encode('utf-8'))

    # 从socket中获得控制码信息
    def __recv_code(self,connection):
        msg_code = connection.recv(1024).decode()
        return msg_code

    # 获得模型路径对应的文件大小
    def __get_model_size(self,model_file_name):
        file_dir = model_file_name
        file_size = os.path.getsize(file_dir)
        return file_size
    
    """
    Public Method
    """
    #开启监听
    def listen(self):
        self.sock.bind((self.host,self.port))
        self.sock.listen(5) #最大连接数
        while True:
            print("监听端口：",(self.host,self.port))
            print("等待连接")
            connect,addr = self.sock.accept()
            print("收到连接请求：", addr)    
            message = self.__recv_code(connect)
            print("请求码: ",message)
            #根据请求码处理请求
            if message=='1001':
                print('请求连接')
                self.__init_paticipant(connect)
            elif message=='1002':
                print('请求模型参数')
                self.__send_model(connect)
                print('模型已发送')
            elif message=='1003':
                print('模型确认')
                #self.__send_model_check(connect)
                #print('回复确认',self.model_status)
            elif message=='1004':
                print('回传梯度')
                #bug?
                gradient_info = self.__recv_gradient(connect)
                self.__params.update_gradient(gradient_info)
                self.__params.validate_current_model()
                self.__params.current_model_accepted(save_dir=self.current_model_dir)
                print('模型更新成功')
            else:
                print("控制码错误",message)
            connect.close()

if __name__ == '__main__':
    server = Sever()
    #server = SocketSever(path_server+'\Trained_Mnist_Mlp.params')
    #server = SocketSever(path_server+'\\random_init_model_MnistMlp.params')
    server.listen()