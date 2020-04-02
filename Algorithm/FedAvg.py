from mxnet import gluon
from mxnet.gluon import nn
from mxnet import ndarray as nd


class Fed_avg_tool():
    # Federated Averaging 算法
    # from paper Communication-Efficient Learning of Deep Networks from Decentralized Data
    def __init__(self, init_model, ctx, cla):
        # init_model初始化存储权值的模型
        self.model_cnt = 0
        self.merge_model = init_model
        self.__set_model_weight2zero()
        self.__ctx = ctx
        self.cla = cla

    def __set_model_weight2zero(self):
        for layer in self.merge_model:
            try:
                zero = nd.zeros(shape=layer.weight.data().shape,ctx=self.__ctx)
                layer.weight.set_data(zero)
                zero = nd.zeros(shape=layer.bias.data().shape,ctx=self.__ctx)
                layer.bias.set_data(zero)
            except:
                continue
                
    def add_fed_model(self, model):
        add_flag = False
        # 将接收的model weight加入fed_avg model中
        for layer1,layer2 in model,self.merge_model:
            try:
                layer2.weigth.data()[:] += layer1.weight.data()[:]  # 可能会出现bug
                layer2.bias.data()[:] += layer1.weight.data()[:]
                if add_flag is False:
                    add_flag = True
            except:
                continue
        self.model_cnt += 1
        if add_flag:
            print("add fed model success")
        
    def get_averaged_model(self):
        average_flag = False
        for layer in self.merge_model:
            try:
                # 算术平均
                layer.weight.data()[:] = layer.weight.data()[:]/self.model_cnt      
                layer.bias.data()[:] = layer.bias.data()[:]/self.model_cnt
                if average_flag is False:
                    average_flag = True
            except:
                continue
        # 数据初始化
        self.__set_model_weight2zero()
        self.model_cnt=0
        if average_flag:
            return self.merge_model
        else:
            raise Exception("Unknown Error in model averaging")

    
    def chk_cla(self):
        if self.cla == self.model_cnt:
            return True
        else:
            return False