import socket
import Client.client_config as client_config

class Client:
    def __init__(self, client_handler):
        self.handler = client_handler
        self.server_ad = client_config.IP_PORT
        self.Client_sock = socket.socket()

    def __send_code(self,msg_code=""):
        # 创建socket连接，发送控制码
        # socket连接将被保存在self.sock中
        self.client_sock = socket.socket()
        self.client_sock.connect(self.server_ad)
        self.client_sock.send(msg_code.encode('utf-8'))

    def __recv_code(self):
        # 从socket中获得控制码信息
        # 控制码大小为4bit
        msg_code = self.client_sock.recv(4).decode()
        return msg_code





    