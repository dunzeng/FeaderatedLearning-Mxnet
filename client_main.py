from Fed_Client import Client
from Fed_Client import Client_data_handler
from Algorithm import LeNet

model = LeNet.LeNet_()
handler = Client_data_handler.Client_data_handler(model)
clinet_network = Client.Client(handler)
clinet_network.process()