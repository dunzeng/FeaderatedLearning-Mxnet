from Fed_Server import Server_data_handler
from Fed_Server import Server
from Algorithm import CNN
from Algorithm import MLP

#model = CNN.CNN_Model("LeNet")
#shape = (1,1,28,28)
model = MLP.MLP()
shape = (1,28,28)
Serv = Server.Sever(model, input_shape=shape, init_model_randomly=True, FedAvg=True)
Serv.listen()