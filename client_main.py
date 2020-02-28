import Fed_Client

model = None
handler = Fed_Client.Client_data_handler.Client_data_handler(model)
network = Fed_Client.Client.Client(handler)
network.process()