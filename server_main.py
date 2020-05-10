from Fed_Server import Server_data_handler
from Fed_Server import Server
from Algorithm import CNN
from Algorithm import MLP

model = CNN.CNN_Model("LeNet")
shape = (1,1,28,28)
#path = "D:\\Mxnet_FederatedLearning\\Fed_Server\\current_model.params"
path = "D:\\Mxnet_FederatedLearning\\LeNet.params"
#model = MLP.MLP()
#shape = (1,28,28)
Serv = Server.Sever(model, input_shape=shape, init_model_randomly=False, 
                    init_model_path=path, FedAvg=True)
Serv.listen()