
from socket import socket
# 配置信息
import server_config


class Server():

    def __init__(self, handler):
        self.handler = handler
        self.server_socket = socket()
    

    def __send_file(self, connect, model_path):
        # 将model_path指向的文件通过connect.send()发送
        pass


    def message_handler(self):
    # 定义网络响应
        pass

    def listen(self):
        self.server_socket.bind(server_config.IP_PORT)
        pass
    