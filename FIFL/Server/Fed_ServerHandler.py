from Utils import utils
import server_config
import mxnet as mx
from mxnet.gluon import ndarray as nd


class ServerHandler:
    def __init__(self, model):
        self.__net = model
        self.__ctx = utils.try_all_gpus()
        
    def __random_init_model(self):
        # 初始化用户自定义的模型
        self.__net.initialize(mx.init.Xavier(magnitude=2.24),ctx=self.__ctx)
        # Mxnet：神经网络在第一次前向传播时初始化,因此初始化神经网络时需要做一次随机前向传播
        self.__net(nd.random.uniform(shape=server_config.SHAPE,ctx=self.__ctx[0]))

    def __load_init_model(self):    
        self.__net.load_parameters(server_config.InitModelFile, ctx=self.__ctx)

    def save_net2file(self, save_path):
        try:
            print("模型保存路径",save_path)
            self.__net.save_parameters(save_path)
        except:
            raise ValueError("Invalid path %s"&save_path)
    
    def __process_gradient(self):
        pass

    def validata_model(self):
        pass

    def process_data_from_client(self, data):
        pass