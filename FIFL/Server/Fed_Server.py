
from socket import socket
# 配置信息
import server_config
import os
from threading import Thread

class Server():

    def __init__(self, server_handler):
        self.handler = server_handler
        self.server_socket = socket()

    def __send_file(self, connect, file_path):
        # 将model_path指向的文件通过connect.send()发送
        file_size = os.path.getsize(file_path)
        print("Sending file... and its size: ",file_size)
        connect.send(str(file_size).encode('utf-8'))
        sent_size = 0
        f = open(file_path,'rb')
        while sent_size != file_size:
            data = f.read(1024)
            connect.send(data)
            sent_size += len(data)
        f.close()
        print("done...")

    def __send_code(self,connection,msg_code=""):
        # 创建socket连接，发送控制信息，socket连接保存
        connection.send(msg_code.encode('utf-8'))

    def __recv_code(self,connection):
        # 从socket中获得控制码信息
        msg_code = connection.recv(4).decode()
        return msg_code

    def listen(self):
        self.server_socket.bind(server_config.IP_PORT)
        while True:
            print("listening port：",server_config.IP_PORT)
            connect,addr = self.server_socket.accept()
            print("received connect request：{}".format(addr))
            self.message_handler(connect)
    
    def multithread_lisen(self):
        self.server_socket.bind(server_config.IP_PORT)
        self.server_socket.listen(5)
        while True:
            new_sock ,client_info = self.server_socket.accept()
            p = Thread(target=self.message_handler, args=(new_sock,client_info))
            p.start()
    
    def __recieve_file(self, connect):
        # 接受模型
        pass

    def message_handler(self, connect):
    # 定义网络响应函数
    # 由使用者重写
        pass