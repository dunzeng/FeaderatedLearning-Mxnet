from mxnet import gluon
from mxnet.gluon import nn
from mxnet import ndarray as nd


class Fed_avg_tool():
    # Federated Averaging 算法
    # from paper Communication-Efficient Learning of Deep Networks from Decentralized Data
    def __init__(self, init_model):
        # init_model初始化存储权值的模型
        self.model_cnt = 0
        self.merge_model = init_model
        self.__set_model_weight2zero()

    def __set_model_weight2zero(self):
        for layer in self.merge_model:
            try:
                zero = nd.zeros(shape=layer.weight.data().shape)
                layer.weight.set_data(zero)
                zero = nd.zeros(shape=layer.bias.data().shape)
                layer.bias.set_data(zero)
            except:
                pass
                
    def add_fed_model(self, model):
        for layer1,layer2 in model,self.merge_model:
            try:
                layer2.weigth.data()[:] += layer1.weight.data()[:]
                layer2.bias.data()[:] += layer1.weight.data()[:]
            except:
                pass
        self.model_cnt += 1
        
    def get_averaged_model(self, model):
        for layer in self.merge_model:
            try:
                layer.weight.data()[:] = layer.weight.data()[:]/self.model_cnt
                layer.bias.data()[:] = layer.bias.data()[:]/self.model_cnt
            except:
                pass
