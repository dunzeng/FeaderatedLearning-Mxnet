import mxnet as mx
import mxnet.ndarray as nd
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd as ag
import copy

class nn_mlp:
    def __init__(self,ctx,epoch=10):
        self.net = nn.Sequential()
        self.net.add(nn.Dense(1024))
        self.net.add(nn.Dense(128,activation='relu'))
        self.net.add(nn.Dense(64,activation='relu'))
        self.net.add(nn.Dense(10))
        #initialize
        self.epoch = epoch
        self.ctx = ctx
        self.net.initialize(mx.init.Xavier(magnitude=2.24), ctx=self.ctx)
    
    def train(self,train_data):
        metric = mx.metric.Accuracy()
        loss = gluon.loss.SoftmaxCrossEntropyLoss()
        trainer = gluon.Trainer(self.net.collect_params(), 'sgd', {'learning_rate': 0.01})
        for i in range(self.epoch):
            train_data.reset()
            for batch in train_data:
                data = gluon.utils.split_and_load(batch.data[0],ctx_list=self.ctx,batch_axis=0)
                label = gluon.utils.split_and_load(batch.label[0], ctx_list=self.ctx, batch_axis=0)
                outputs = []
                with ag.record():
                    for x,y in zip(data, label):
                        z = self.net(x)
                        t_loss = loss(z,y)  #损失函数
                        t_loss.backward() #函数自动求梯度   y= net(x) (z)
                        outputs.append(z)
                metric.update(label,outputs)
                trainer.step(batch.data[0].shape[0])  #batch.data[0].shape[0]=batchsize  this function is update parameters
            name, acc = metric.get()
            metric.reset()
            print('training acc at epoch %d/%d: %s=%f'%(self.epoch,i, name, acc))
    
    def get_gradient(self):
        pass

    def validation(self,val_data):
        pass

    def get_networks(self):
        return copy.deepcopy(self.net)
    
    def load_file(self,model_dir=""):
        self.net.load_parameters(model_dir,ctx=self.ctx)
        print('load finished')
    
    def save_model(self,model_dir=""):
        self.net.save_parameters(model_dir)
        print('Model saved')

class nn_cnn:
    def __init__(self,ctx,epoch=10):
        pass

    def train(self,train_data):
        pass