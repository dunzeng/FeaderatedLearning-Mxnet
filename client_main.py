from Fed_Client import Client
from Fed_Client import Client_data_handler
from Algorithm import CNN
import time
import random

def single_process(data_idx=0):
    model = CNN.CNN_Model('LeNet')
    train_data = {}
    data = "E:\\PythonProjects\\Mxnet_FederatedLearning\\Fed_Client\\FedAvg_data\\train_data" + str(data_idx) + ".npy"
    label = "E:\\PythonProjects\\Mxnet_FederatedLearning\\Fed_Client\\FedAvg_data\\train_label" + str(data_idx) + ".npy"
    train_data['data'] = data
    train_data['label'] = label
    handler = Client_data_handler.Client_data_handler(model,input_shape=(1,1,28,28),train_data_path=train_data)
    clinet_network = Client.Client(handler)
    clinet_network.process()
    time.sleep(1.5)


def fed_process():
    # FedAvg
    client = []
    cla = 10
    while len(client) < cla:
        num = random.randint(0,99)
        if num not in client:
            client.append(num)
    print(client)
    for idx in client:
        single_process(idx)

if __name__=="__main__":
    for i in range(10):
        single_process()


    
    