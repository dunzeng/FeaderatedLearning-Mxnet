
from socket import socket
# 配置信息
from Server import server_config
import os
from threading import Thread
from Utils.network_service import serve

class Fed_Server(serve):
    def __init__(self, server_handler):
        super(Fed_Server,self).__init__()
        self.handler = server_handler

    def listen(self):
        self.sock.bind(server_config.IP_PORT)
        self.sock.listen(5)
        while True:
            print("listening port：",server_config.IP_PORT)
            connect,addr = self.sock.accept()
            print("received connect request：{}".format(addr))
            stop_flag = self.message_handler(connect)
            if stop_flag is True:
                break
    
    def multithread_lisen(self):
        self.sock.bind(server_config.IP_PORT)
        self.sock.listen(5)
        while True:
            new_sock ,client_info = self.sock.accept()
            p = Thread(target=self.message_handler, args=(new_sock,client_info))
            p.start()

    def message_handler(self, connect):
    # 定义网络响应函数
    # 由使用者重写
        return False