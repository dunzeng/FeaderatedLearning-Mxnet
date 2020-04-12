import sys
path_base = "E:\\PythonProjects\\Mxnet_FederatedLearning"
sys.path.append(path_base)
import mxnet as mx
from socket import socket
import os
import pickle
from Fed_Server.Server_data_handler import Server_data_handler
import json
from Tools import utils
#from Tools.log import log
import time
#from multiprocessing import Process
from threading import Thread
mx.random.seed(int(time.time()))
from Fed_Server.Server_log import server_log
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
        # 保存模型至本地
        self.data_handler.save_current_model2file(self.update_model_path)
        # 网络连接采用TCP协议
        self.server_socket = socket()
        
        # 训练模式
        self.train_mode = json_data['train_mode']
        # log类
        self.log = server_log(path_base + "\\Fed_Server\\log")
        # 通信轮数
        self.communicate_round = 0


    def __get_val_data(self):
        mnist = mx.test_utils.get_mnist()
        val_data = [mnist['test_data'],mnist['test_label']]
        return val_data

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

    def __recv_data(self,connection):
        # 接收参与者回传梯度
        # ***下一阶段考虑实现由服务器求梯度下传Client更新模型
        gradient_dict = utils.recv_class(connection)
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
    
    def __send_server_param(self,connection):
        # train_mode learning_rate input_shape
        param_dict = self.data_handler.get_param_dict()
        param_dict["train_mode"] = self.train_mode
        print("发送参数 ",param_dict)
        utils.send_class(connection,param_dict)
         
    def message_handler(self, new_sock, clinet_info):
        # 多线程响应
        # 线程同步机制未添加
        print("客户端{}已经连接".format(clinet_info))
        # 多线程处理Client信息
        message = self.__recv_code(new_sock)
        #根据请求码处理请求
        if message=='1001':
            # Client池维护
            # 留空控制码
            print('请求连接')
        elif message=='1002':
            # 发送模型
            print("******Client端请求模型******")
            self.__send_model(new_sock)
            print('---Server Model已发送---\n\n')
        elif message=='1003':
            # 参数同步
            print("******Client端请求系统参数******")
            self.__send_server_param(new_sock)
            print("---系统参数已发送---\n\n")
        elif message=='1004':
            # 接收Client信息
            print('******Client端请求上传信息******')
            data_from_client = self.__recv_data(new_sock)
            self.data_handler.process_data_from_client(data_from_client,mode=self.train_mode)
            self.data_handler.validate_current_model()
            self.data_handler.current_model_accepted(self.update_model_path)  # 覆盖更新
            print('---模型更新成功---\n\n')
        else:
            print("Control Code Error ",message)
        new_sock.close()
    
    def multithread_lisen(self):
        # 监听
        # 多线程后台执行
        self.server_socket.bind(address=(self.host,self.port))
        self.server_socket.listen(5)
        while True:
            new_sock ,client_info = self.server_socket.accept()
            p = Thread(target=self.message_handler, args=(new_sock,client_info))
            p.start()
    
    def listen(self):
        # C/S架构
        # 网络监听 根据不同的控制码 服务端执行不同操作
        # Client连接->下传模型->接收梯度->更新模型
        self.server_socket.bind((self.host,self.port))
        self.server_socket.listen(5) #最大连接数
        while True:
            print("监听端口：",(self.host,self.port))
            connect,addr = self.server_socket.accept()
            # 接收到请求 单次处理一个连接请求
            print("收到连接请求：{}".format(addr))    
            message = self.__recv_code(connect)
            #根据请求码处理请求
            if message=='1001':
                # Client池维护
                # 留空控制码
                print('请求连接')
            elif message=='1002':
                # 发送模型
                print("******Client端请求模型******")
                self.__send_model(connect)
                print('---Server Model已发送---\n\n')
            elif message=='1003':
                # 参数同步
                print("******Client端请求系统参数******")
                self.__send_server_param(connect)
                print("---系统参数已发送---\n\n")
            elif message=='1004':
                # 接收Client信息
                print('******Client端请求上传信息******')
                data_from_client = self.__recv_data(connect)
                self.data_handler.process_data_from_client(data_from_client,mode=self.train_mode)
                acc = self.data_handler.validate_current_model()
                self.data_handler.current_model_accepted(self.update_model_path)
                print('---模型更新成功---\n\n')
                self.log.record_acc(acc)
                self.log.add_cummu_round()
            elif message=='6666':
                self.log.record_to_file()
                break
            else:
                print("Control Code Error ",message)
            connect.close()