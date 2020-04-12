import socket


client_sock = socket.socket()
client_sock.connect(("localhost",8080))
msg_code = "6666"
client_sock.send(msg_code.encode('utf-8'))