# FIFL demo 
# Federated Averaging : client端
import os,sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from mxnet.gluon import nn
from Client.Fed_ClientHandler import ClientHandler
from Client.Fed_Client import Client
from Client import client_config

# 定义神经网络模型结构
model = nn.Sequential()
model.add(nn.Dense(128,activation='relu'),
        nn.Dense(64,activation='relu'),
        nn.Dense(10))

# 定义逻辑层算法
class my_handler(ClientHandler):
    def __init__(self, model):
        super(my_handler,self).__init__(model)


# 定义网络层协议
class my_client(Client):
    def __init__(self, handler):
        super(my_client,self).__init__(handler)

        self.server_ad = client_config.IP_PORT
        self.model_path = client_config.ModelPath

        self.batch_size = None
        self.learning_rate = None
        self.epoch = None
        self.mode = None
        self.ask_params() #请求梯度

    def ask_params(self):
        self.send_code(self.server_ad, "1001")
        params = self.recv_class(self.sock)
        print("Client: 同步参数 ", params)
        self.batch_size = params['batch_size']
        self.learning_rate = params['lr']
        self.epoch = params['epoch']
        self.mode = params['mode']
        self.sock.close()

    def ask_model(self):
        self.send_code(self.server_ad, "1002")
        self.recv_file(self.model_path)
        self.sock.close()
        self.handler.load_model(self.model_path)
        
        
    def send_model(self):
        self.send_code(self.server_ad, "1003")
        net = self.handler.get_model()
        self.send_class(self.sock,net)
        self.sock.close()

    def process(self):
        # Client端流程 
        # 初始化参数->请求模型->加载模型->训练->梯度回传
        # 考虑不同算法 朴素Fed,FedAvg回传信息时的处理
        #self.ask_params()
        self.ask_model()
        self.handler.local_train(self.epoch, self.learning_rate, self.batch_size)
        self.send_model()

if __name__ == "__main__":
    handler = my_handler(model)
    client = my_client(handler)
    client.process()
