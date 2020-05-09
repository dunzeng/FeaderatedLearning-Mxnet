
from socket import socket
# 配置信息
import server_config
import os

class Server():

    def __init__(self, handler):
        self.handler = handler
        self.server_socket = socket()
    

    def __send_file(self, connect, model_path):
        # 将model_path指向的文件通过connect.send()发送
        file_size = os.path.getsize(model_path)
        print("发送模型大小：",file_size)
        connect.send(str(file_size).encode('utf-8'))
        sent_size = 0
        f = open(model_path,'rb')
        while sent_size != file_size:
            data = f.read(1024)
            connect.send(data)
            sent_size += len(data)
        f.close()
    
    def __recieve_file(self, connect, ):
        # 
        pass

    def message_handler(self, connect):
    # 定义网络响应
        pass

    def listen(self):
        self.server_socket.bind(server_config.IP_PORT)
        while True:
            print("监听端口：",server_config.IP_PORT)
            connect,addr = self.server_socket.accept()
            print("收到连接请求：{}".format(addr))
            self.message_handler(connect)
    