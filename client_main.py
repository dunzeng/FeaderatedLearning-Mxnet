from Fed_Client import Client
from Fed_Client import Client_data_handler
from Algorithm import CNN

model = CNN.CNN_Model('LeNet')
handler = Client_data_handler.Client_data_handler(model,input_shape=(1,1,28,28))
clinet_network = Client.Client(handler)
clinet_network.process()