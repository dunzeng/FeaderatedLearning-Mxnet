# FIFL demo 
# Federated Averaging : server端
# 主目录路径
import os,sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from mxnet.gluon import nn
from Server import server_config
from Server.Fed_Server import Fed_Server
from Server.Fed_ServerHandler import ServerHandler

# 定义神经网络模型结构
model = nn.Sequential()
model.add(nn.Dense(128,activation='relu'),
        nn.Dense(64,activation='relu'),
        nn.Dense(10))

# 定义逻辑层算法
class my_handler(ServerHandler):
    def __init__(self, model):
        super(my_handler, self).__init__(model)
        

class my_server(Fed_Server):
    def __init__(self, handler):
        super(my_server, self).__init__(handler)
        self.current_model_path = server_config.ModelSavePath
        self.val_data = None

    def __get_train_params(self):
        train_params = {}
        train_params['epoch'] = server_config.Epoch
        train_params['lr'] = server_config.LR
        train_params['batch_size'] = server_config.BatchSize
        train_params['mode'] = server_config.Mode
        return train_params

    def get_val_data(self):
        pass

    def message_handler(self, connect):
        # 重写响应函数
        # 1001 请求参数 1002 请求模型 1003 回传信息
        code = self.recv_code(connect)
        if code == "1001":
            params = self.__get_train_params()
            self.send_class(connect, params)
        elif code == "1002":
            self.send_file(connect, self.current_model_path)
        elif code == "1003":
            data = self.recv_class(connect)
            self.handler.process_data_from_client(data)
            self.handler.save_net2file(self.current_model_path) #保存模型
            #self.handler.validata_model(self.val_data)
        else:
            print("Invalid Communicate Code: ",code)
        connect.close()

if __name__ == "__main__":
    handler = my_handler(model)
    server = my_server(handler)
    server.listen()