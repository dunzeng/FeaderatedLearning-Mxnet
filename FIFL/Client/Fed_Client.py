
import socket
from Utils.network_service import serve
from Client import client_config


class Client(serve):
    def __init__(self, client_handler):
        super(Client, self).__init__()
        self.handler = client_handler
        self.server_ad = client_config.IP_PORT

    