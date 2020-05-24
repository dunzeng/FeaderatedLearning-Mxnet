from socket import socket
import pickle
from tqdm import tqdm
import time
import os

class client_serve():
    def __init__(self):
        self.sock = socket()

    def send_code(self, msg_code):
        pass

    def recv_class(self, data):
        pass

    def recv_file(self, data):
        pass

    def send_class(self, data):
        pass

class serve():
    def __init__(self):
        self.sock = socket()

    def __cut_bytes(self, data, block_size):
        # 将data切割为大小为block_size每块的列表
        size = len(data)
        ret = []
        idx = 0
        while True:
            if idx+block_size<size:
                ret.append(data[idx:idx+block_size])
            else:
                ret.append(data[idx:])
                break
            idx += block_size 
        return ret

    def send_class(self, connect, class_data):
        print("Socket: 发送内存数据")
        # 利用pickle模块将data序列化，并通过connect发送
        # connect参数为socket连接对象
        data = pickle.dumps(class_data)
        data_list = self.__cut_bytes(data,1024)
        list_size = len(data_list)
        msg = str(list_size).encode('utf-8')
        connect.send(msg)
        time.sleep(0.2)
        for i in tqdm(range(list_size),desc="Sending Data"):
            connect.send(data_list[i])
        print("Socket: 结束")

    def recv_class(self, connect):
        print("Socket: 接受数据")
        # 与send_class成对使用，从socket连接中接收数据
        # 1024bit为1Byte
        tmp_data = connect.recv(1024)
        try:
            block_size = int(tmp_data.decode())
        except:
            print("Error: %s"%(str(tmp_data)))
            raise Exception("Decode Error")
        data = bytes()
        for _ in tqdm(range(block_size),desc="Recving Data"):
            data_slice = connect.recv(1024)
            data += data_slice
        class_info = pickle.loads(data)
        print("Socket: 结束")
        return class_info
        
    def send_code(self, addr, msg_code):
        # client请求socket连接，发送控制码
        # socket连接将被保存在self.sock中
        self.sock = socket()
        self.sock.connect(addr)
        self.sock.send(msg_code.encode('utf-8'))
        print("Socket: sending control code ",msg_code)

    def recv_code(self, connect):
        # Server端接收控制码
        # 从socket中获得控制码信息
        # 控制码大小为4bit
        msg_code = connect.recv(4).decode()
        print("Socket: receive control code ",msg_code)
        return msg_code

    def send_file(self, connect, file_path):
        # Server端下发模型
        print("Socket: 发送文件")
        # 将model模型文件发送至Client
        # 从文件中读取发送模型
        file_size = os.path.getsize(file_path)
        connect.send(str(file_size).encode('utf-8'))
        sent_size = 0
        f = open(file_path,'rb')
        while sent_size != file_size:
            data = f.read(1024)
            connect.send(data)
            sent_size += len(data)
        f.close()
        print("Socket: 结束")

    def recv_file(self, save_path):
        # Client端接收模型文件
        print("Socket: 接收文件")
        # 接收模型
        # 下载模型并写入文件
        file_size = int(self.sock.recv(1024).decode())
        has_recv = 0
        f = open(save_path,'wb')
        while has_recv != file_size:
            data = self.sock.recv(1024)
            f.write(data)
            has_recv += len(data)
        f.close()
        print("Socket: 结束")