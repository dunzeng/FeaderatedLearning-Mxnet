from Fed_Client import Client
from Fed_Client import Client_data_handler
from Algorithm import CNN
import time
def single_process(data_idx=0):
    model = CNN.CNN_Model('LeNet')
    train_data = {}
    data = "E:\\PythonProjects\\Mxnet_FederatedLearning\\Fed_Client\\local_data\\train_data_600_" + str(data_idx) + ".npy"
    label = "E:\\PythonProjects\\Mxnet_FederatedLearning\\Fed_Client\\local_data\\train_label_600_" + str(data_idx) + ".npy"
    train_data['data'] = data
    train_data['label'] = label
    handler = Client_data_handler.Client_data_handler(model,input_shape=(1,1,28,28),train_data_path=train_data)
    clinet_network = Client.Client(handler)
    clinet_network.process()

if __name__=="__main__":
    
    for i in range(10):
        single_process(i)
        time.sleep(5)
    
    #single_process()