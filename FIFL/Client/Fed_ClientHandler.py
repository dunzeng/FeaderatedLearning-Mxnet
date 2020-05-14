from Utils import utils
import client_config
import mxnet as mx
from mxnet.gluon import ndarray as nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet import gluon
from copy import deepcopy
class ClientHandler:
    def __init__(self, model):
        self.__net = model
        self.__ctx = utils.try_all_gpus()
        # 梯度列表
        self.grad_dict = {"weight":[],"bias":[]}
        
    def __init_model_randomly(self):
        # 随机初始化用户自定义的模型
        self.__net.initialize(mx.init.Xavier(magnitude=2.24),ctx=self.__ctx)
        self.__net(nd.random.uniform(shape=client_config.Data_Shape,ctx=self.__ctx[0]))

    def __load_model(self, model_path):
        self.__net.load_parameters(model_path,ctx=self.__ctx)

    def __init_gradient_list(self):
        self.grad_dict['weight'].clear()
        self.grad_dict['bias'].clear()
        for layer in self.__net:
            try:
                shape_w = layer.weight.data().shape
                shape_b = layer.bias.data().shape
            except:
                continue
            self.grad_dict['weight'].append(nd.zeros(shape=shape_w,ctx=self.__ctx[0]))
            self.grad_dict['bias'].append(nd.zeros(shape=shape_b,ctx=self.__ctx[0]))

    def __updata_local_model(self):
        # 收集梯度信息
        batch_size = client_config.Batch_size
        idx = 0
        for layer in self.__net:
            try:
                grad_w = layer.weight.data().grad
                self.grad_dict["weight"][idx] += grad_w/batch_size
                grad_b = layer.bias.data().grad
                self.grad_dict["bias"][idx] += grad_b/batch_size
                idx += 1
            except:
                continue
    
    def __train_data_loader(self, batch_size):
        # pickle
        path = client_config.Train_Data_Path
        data,label = None,None
        train_data = mx.io.NDArrayIter(data,label,batch_size=batch_size,shuffle=True)
        return train_data

    def local_train(self, epoch, learning_rate, batch_size):
        # 训练数据
        train_data = self.__train_data_loader(batch_size)
        # 训练组件
        smc_loss = gluon.loss.SoftmaxCrossEntropyLoss()
        metric = mx.metric.Accuracy()
        trainer = gluon.Trainer(self.__net.collect_params(), 'sgd', {'learning_rate': learning_rate})
        # 开始训练
        for i in range(epoch):
            train_data.reset()
            for batch in train_data:
                data = gluon.utils.split_and_load(batch.data[0],ctx_list=self.__ctx,batch_axis=0)
                label = gluon.utils.split_and_load(batch.label[0], ctx_list=self.__ctx, batch_axis=0)
                outputs = []
                with ag.record():
                    for x,y in zip(data,label):
                        z = self.__net(x)
                        loss = smc_loss(z,y)
                        loss.backward()
                        outputs.append(z)
                metric.update(label,outputs)
                trainer.step(batch.data[0].shape[0])
            name, acc = metric.get()
            metric.reset()
            print('training acc at epoch %d/%d: %s=%f'%(i+1,epoch, name, acc))
            if acc>= 1:
                # 过拟合 结束本地训练
                break

    def get_model(self):
        # 获得模型 上层调用
        return deepcopy(self.__net)
    
    def get_gradient(self):
        # 获得梯度 上层调用
        return deepcopy(self.grad_dict)