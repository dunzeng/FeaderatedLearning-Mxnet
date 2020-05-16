# FIFL demo 
# Federated Averaging : client端


from mxnet.gluon import nn
from Client.Fed_ClientHandler import ClientHandler
from Client.Fed_Client import Client


# 定义神经网络模型结构
model = nn.Sequential()
model.add(nn.Dense(128,activation='relu'),
        nn.Dense(64,activation='relu'),
        nn.Dense(10))

# 定义逻辑层算法
class my_handler(ClientHandler):
    def __init__(self, model):
        super.__init__(model)

class my_client(Client):
    def __init(self, model):
        super.__init__(model)


    def process(self):
        pass

if __name__ == "__main__":
    handler = my_handler(model)
    client = Client(handler)
