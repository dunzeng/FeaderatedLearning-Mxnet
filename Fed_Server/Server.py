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
from Fed_Server.Server_data_handler import Server_data_handler
import json
import Tools

"""
网络服务器：
维护网络通信，与Client交流
与后台数据处理类进行信息交换
"""
class Sever():
    def __init__(self,server_data_handler):
        # server_data_handler由开发者初始化作为成员参数传入Server类
        # 从Json文件中读取系统配置
        with open(path_base+"\\Fed_Server\\server_config.json",'r') as f:
            json_data = json.load(f)
        self.port = json_data['port']
        self.host = json_data['host']
        # 存储更新的模型
        self.update_model_path = json_data['update_model_path']
        # 初始化Server端模型
        # data_handler自动初始化模型 并将模型储存至update_model_path指向的文件
        self.data_handler = server_data_handler
        self.data_handler.save_current_model2file(self.update_model_path)
        # 网络连接采用TCP协议
        self.sock = socket.socket()
    
    def __send_model(self,connection,model_path=""):
        # 将model模型文件发送至Client
        # 从文件中读取发送模型
        file_size = self.__get_model_size(self.update_model_path)
        file_dir = self.update_model_path
        print("发送模型大小：",file_size)
        connection.send(str(file_size).encode('utf-8'))
        sent_size = 0
        f = open(file_dir,'rb')
        while sent_size != file_size:
            data = f.read(1024)
            connection.send(data)
            sent_size += len(data)
        f.close()

    def __recv_gradient(self,connection):
        # 接收参与者回传梯度
        # ***下一阶段考虑实现由服务器求梯度下传Client更新模型
        gradient_dict = Tools.utils.recv_class(connection)
        return gradient_dict

    def __send_code(self,connection,msg_code=""):
        # 创建socket连接，发送控制信息，socket连接保存
        connection.send(msg_code.encode('utf-8'))

    def __recv_code(self,connection):
        # 从socket中获得控制码信息
        msg_code = connection.recv(4).decode()
        return msg_code

    def __get_model_size(self,model_file_name):
        # 获得模型路径对应的文件大小
        file_dir = model_file_name
        file_size = os.path.getsize(file_dir)
        return file_size
    
    def listen(self):
        # C/S架构
        # 网络监听 根据不同的控制码 服务端执行不同操作
        # Client连接->下传模型->接收梯度->更新模型
        self.sock.bind((self.host,self.port))
        self.sock.listen(5) #最大连接数
        while True:
            print("监听端口：",(self.host,self.port))
            connect,addr = self.sock.accept()
            # 接收到请求 单次处理一个连接请求
            print("收到连接请求：", addr)    
            message = self.__recv_code(connect)
            #根据请求码处理请求
            if message=='1001':
                print('请求连接')
            elif message=='1002':
                print('请求Server下传模型')
                self.__send_model(connect)
                print('Server Model已发送')
            elif message=='1004':
                print('Client请求上传模型')
                gradient_info = self.__recv_gradient(connect)
                self.data_handler.update_gradient(gradient_info)
                self.data_handler.validate_current_model()
                self.data_handler.save_current_model2file(self.update_model_path)
                print('模型更新成功')
            else:
                print("Control Code Error ",message)
            connect.close()