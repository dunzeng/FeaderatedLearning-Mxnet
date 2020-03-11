from Fed_Server import Server_data_handler
from Fed_Server import Server
from Algorithm import LeNet

model = LeNet.LeNet_(activation='relu')
handler = Server_data_handler.Server_data_handler(model)
Serv = Server.Sever(handler)

Serv.listen()