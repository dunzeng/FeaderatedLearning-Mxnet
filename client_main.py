from Fed_Client.Client import Client
from Algorithm import CNN
from Algorithm import MLP
import time
import random

def single_process(data_idx=0):
    model = CNN.CNN_Model('LeNet')
    shape = (1,1,28,28)
    #model = MLP.MLP()
    #shape = (1,28,28)
    train_data = {}
    data = "E:\\PythonProjects\\Mxnet_FederatedLearning\\Fed_Client\\FedAvg_data_random\\train_data" + str(data_idx) + ".npy"
    label = "E:\\PythonProjects\\Mxnet_FederatedLearning\\Fed_Client\\FedAvg_data_random\\train_label" + str(data_idx) + ".npy"
    train_data['data'] = data
    train_data['label'] = label
    clinet_network = Client(model, input_shape=shape, train_data=train_data)
    clinet_network.process()
    time.sleep(0.5)

def fed_process():
    # FedAvg
    client = [x for x in range(100)]
    random.shuffle(client)
    for id in range(100):
        single_process(id)

if __name__=="__main__":
    #for i in range(5):
    #    single_process()

    for _ in range(1000):
        fed_process()
    
    #single_process(0)

    
    