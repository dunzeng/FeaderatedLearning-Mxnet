# LSTM 下一词预测
# mxnet.gluon.rnn.LSTM
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd as ag
from mxnet import ndarray as nd
from mxnet.gluon import loss as gloss, nn, rnn
import time
context = [mx.gpu()]

class RNNModel(gluon.Block):
    # 循环神经网络库
    def __init__(self, mode, vocab_size, embed_dim, hidden_dim, num_layers, dropout=0.5, **kwargs):
        super(RNNModel,self).__init__(**kwargs)
        with self.name_scope():
            self.drop = nn.Dropout(dropout)
            self.encoder = nn.Embedding(vocab_size, embed_dim, weight_initializer=mx.init.Uniform(0.1))

            if mode == 'rnn_relu':
                self.rnn = rnn.RNN(hidden_dim, num_layers, activation='relu', dropout=dropout, input_size=embed_dim)
            elif mode == 'rnn_tanh':
                self.rnn = rnn.RNN(hidden_dim, num_layers, dropout=dropout, input_size=embed_dim)
            elif mode == 'lstm':
                self.rnn = rnn.LSTM(hidden_dim, num_layers, dropout=dropout, input_size=embed_dim)
            elif mode == 'gru':
                self.rnn = rnn.GRU(hidden_dim, num_layers, dropout=dropout, input_size=embed_dim)
            else:
                raise ValueError("Invalid mode %s. Options are rnn_relu, rnn_tanh, lstm, and gru"&mode)

            self.decoder = nn.Dense(vocab_size, in_units=hidden_dim)
            self.hidden_dim = hidden_dim
    
    def forward(self, inputs, state):
        emb = self.drop(self.encoder(inputs))
        output, state = self.rnn(emb, state)
        output = self.drop(output)
        decoded = self.decoder(output.reshape((-1, self.hidden_dim)))
        return decoded, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)
 
    def train(self, train_data):
        pass

    
def batchify(data, batch_size):
    # 数据形状 (num_batches, batch_size)
    num_batches = data.shape[0] // batch_size
    data = data[:num_batches * batch_size]
    data = data.reshape((batch_size, num_batches)).T
    return data

model_name = 'rnn_rlu'
embed_dim = 100
hidden_dim = 100
num_layers = 2
lr = 1.0
clipping_norm = 0.2
epochs =1 
batch_size = 32
num_steps = 5
dropout_rate = 0.2
eval_period = 500
vocab_size = 10000

model = RNNModel(model_name, vocab_size, embed_dim, hidden_dim, num_layers, dropout_rate)
model.collect_params().initialize(mx.init.Xavier(),ctx=context) #ctx
trainer = gluon.Trainer(model.collect_params(),'sgd',{'learning_rate':lr, 'momentum':0, 'wd':0})
loss = gluon.loss.SoftmaxCrossEntropyLoss()

def get_batch(source, i):
    seq_len = min(num_steps, source.shape[0]-1-i)
    data = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len]
    return data, target.reshape((-1,))

def detach(state):
    if isinstance(state, (tuple, list)):
        state = [i.detach() for i in state]
    else:
        state = state.detach()
    return state

def model_eval(data_source, model):
    total_L = 0.0
    ntotal = 0
    hidden = model.begin_state(func=mx.nd.zeros, batch_size=batch_size, ctx=context)

    for i in range(0, data_source.shape[0]-1, num_steps):
        data, target = get_batch(data_source, i)
        output, hidden = model(data, hidden)
        L = loss(output, target)
        total_L += mx.nd.sum(L).asscalar()
        ntotal += L.__size
    
    return total_L / ntotal

def train():
    for epoch in range(epochs):
        total_L = 0,0
        start_time = time.time()
        hidden = model.begin_state(func=mx.nd.zeros, batch_size=batchsize, ctx=context)
        
        for ibatch,i in enumerate(range(0, train_data.shape[0]-1, num_steps)):
            data,target = get_batch(train_data, i)
            #从计算图分离隐含状态
            hidden = detach(hidden)
            with autograd.record():
                output, hidden = model(data, hidden)
                L = loss(output, target)
                L.backward()
            grad = [i.grad(context) for i in model.collect_params().values()]
            #梯度裁剪
            gluon.utils.clip_global_norm(grads, clipping_norm * num_steps * batch_size)
            trainer.step(batch_size)
            total_L += mx.nd.sum(L).asscalar()
        


