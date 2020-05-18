from Utils import utils
from Server import server_config
import mxnet as mx
from mxnet import gluon
from mxnet import ndarray as nd


class ServerHandler:
    def __init__(self, model, init_model_randomly=True):
        self.__net = model
        self.__ctx = utils.try_all_gpus()
        if init_model_randomly is True:
            self.__random_init_model()
        else:
            self.__load_init_model()
        self.save_net2file(server_config.ModelSavePath)

    def __random_init_model(self):
        print("initializing networks randomly...")
        # 初始化用户自定义的模型
        self.__net.initialize(mx.init.Xavier(magnitude=2.24),ctx=self.__ctx)
        # Mxnet：神经网络在第一次前向传播时初始化,因此初始化神经网络时需要做一次随机前向传播
        self.__net(nd.random.uniform(shape=server_config.SHAPE,ctx=self.__ctx[0]))

    def __load_init_model(self):
        print("loading networks from file...")    
        self.__net.load_parameters(server_config.InitModelFile, ctx=self.__ctx)
    
    def save_net2file(self, save_path):
        try:
            print("saving networks to file: ",save_path)
            self.__net.save_parameters(save_path)
        except:
            raise ValueError("Invalid path %s"&save_path)
    
    def validata_model(self, val_data, batch=100):
        # 给定数据集测试模型性能
        # 评估当前模型准确率
        val_data = mx.io.NDArrayIter(val_data['data'],val_data['label'],batch_size=batch)
        print("validating networks...")
        for batch in val_data:
            data = gluon.utils.split_and_load(batch.data[0],ctx_list=self.__ctx,batch_axis=0)
            label = gluon.utils.split_and_load(batch.label[0],ctx_list=self.__ctx,batch_axis=0)
            outputs = []
            metric = mx.metric.Accuracy()  
            for x in data:
                outputs.append(self.__net(x))
            metric.update(label,outputs)
        _,acc = metric.get()
        print('validation accuracy: %f'%(acc))
        return acc
    
    def weight_process(self, weight_data):
        self.__net = weight_data
    
    def gradient_process(self, gradient_dict):
        # 将梯度信息更新到全局模型上
        # 由Client回传的梯度信息 更新Server模型
        lr = server_config.LR
        gradient_w = gradient_dict['weight']
        gradient_b = gradient_dict['bias']
        idx = 0
        update_flag = False
        for layer in self.__net:
            try:
                layer.weight.data()[:] -= gradient_w[idx].as_in_context(layer.weight.data().context) * lr
                layer.bias.data()[:] -= gradient_b[idx].as_in_context(layer.bias.data().context) * lr
                if update_flag is False:
                    update_flag = True
            except:
                continue
            idx += 1
        if update_flag:
            print("-gradient successfully updated-")
        else:
            raise Warning("-oops! gradient failure-")
    
    def defined_process(self,data):
        pass

    def process_data_from_client(self, data):
        if server_config.Mode == 'Weight':
            self.__net = data
        elif server_config.Mode == 'Gradient':
            self.gradient_process(data)
        elif server_config.Mode == 'Defined':
            self.defined_process(data)
        else:
            raise ValueError("Invalid Mode %s. Options are Weight, Gradient and Defined "&server_config.Mode)