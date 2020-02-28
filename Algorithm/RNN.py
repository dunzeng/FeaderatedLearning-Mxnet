# LSTM 下一词预测
# mxnet.gluon.rnn.LSTM
import sys
path_base = "E:\PythonProjects\Mxnet_FederatedLearning"
sys.path.append(path_base)
import Tools
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd as ag
from mxnet import ndarray as nd
from mxnet.gluon import loss as gloss, nn, rnn
import time
import zipfile
import math

context = [mx.gpu()]
num_epochs, batch_size, lr, clipping_theta = 250, 32, 1e2, 1e-2

class RNN_Model(gluon.Block):
    # 循环神经网络库
    def __init__(self, mode, vocab_size, embed_dim, hidden_dim, num_layers, dropout=0.5, **kwargs):
        """
        mode: RNN网络类型
        vocab_size: 数据集不同字符的数目
        embed_dim: 编码器神经元数目 输入数据特征数目
        hidden_dim: 隐藏状态特征数
        num_layers: 循环神经网络层数
        dropout: 随机失活
        
        stata shape: (隐藏层个数, 批量大小, 隐藏单元个数)
        input shape: (时间步数, 批量大小, 输入个数)
        """
        super(RNN_Model,self).__init__(**kwargs)
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

    def begin_state(self, batch_size, ctx):
        # 初始状态shape (隐藏层个数, 批量大小, 隐藏单元个数)
        return self.rnn.begin_state(batch_size=batch_size,ctx=context)
    
    """
    def grad_clipping(self, params, theta, ctx):
        """Clip the gradient."""
        if theta is not None:
            norm = nd.array([0], ctx)
            for param in params:
                norm += (param.grad ** 2).sum()
            norm = norm.sqrt().asscalar()
            if norm > theta:
                for param in params:
                    param.grad[:] *= theta / norm
    
    
    def train(self, train_data, num_epochs, num_steps, clipping_norm, pred_period):
        #input:
        trainer = gluon.Trainer(self.rnn.collect_params(),'sgd',{'learning_rate':lr, 'momentum':0, 'wd':0})
        loss = gluon.loss.SoftmaxCrossEntropyLoss()
        self.rnn.initialize(ctx=context, force_reinit=True, init=mx.init.Xavier())
        for epoch in range(num_epochs):
            L_sum = 0.0
            start_time = time.time()
            #数据迭代器
            #data_iter = Tools.utils.data_iter_consecutive(corpus_indices, batch_size, num_steps, context)
            #RNN初始状态
            state = self.begin_state(batch_size=batch_size, ctx=context)
            for X, Y in train_data:
                for s in state:
                    s.detach()
                with ag.record():
                    output, state = self.rnn(X, state)
                    y = Y.T.reshape((-1,))
                    L = loss(output, Y)
                L.backward()
                grads = [i.grad(context) for i in model.collect_params().values()]
                #梯度裁剪
                #当前梯度是整个批量的梯度
                #因此要将clipping_norm乘以num_steps和batch_size
                gluon.utils.clip_global_norm(grads, clipping_norm * num_steps * batch_size)
                trainer.step(batch_size)
                L_sum += L.asscalar() * Y.size
                n += y.size
            if (epoch + 1) % pred_period == 0:
                print('epoch %d, perplexity %f, time %.2f sec' % (epoch + 1, math.exp(L_sum / n), time.time() - start_time))

    
    def predict_rnn_gluon(self, prefix, num_chars, vocab_size, ctx, idx_to_char, char_to_idx):
        # inputs: prefix 需要预测的输入 num_chars 预测字的个数 vocab_size 语料库字符数
        # 使用model的成员函数来初始化隐藏状态
        state = self.rnn.begin_state(batch_size=1, ctx=ctx)
        output = [char_to_idx[prefix[0]]]
        for t in range(num_chars + len(prefix) - 1):
            X = nd.array([output[-1]], ctx=ctx).reshape((1, 1))
            (Y, state) = self.rnn(X, state)  # 前向计算不需要传入模型参数
            if t < len(prefix) - 1:
                output.append(char_to_idx[prefix[t + 1]])
            else:
                output.append(int(Y.argmax(axis=1).asscalar()))
        return ''.join([idx_to_char[i] for i in output])

def load_data_jay_lyrics():
    """Load the Jay Chou lyric data set (available in the Chinese book)."""
    #返回 四元组
    with zipfile.ZipFile('../data/jaychou_lyrics.txt.zip') as zin:
        with zin.open('jaychou_lyrics.txt') as f:
            corpus_chars = f.read().decode('utf-8')
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    corpus_chars = corpus_chars[0:10000]
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)
    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    return corpus_indices, char_to_idx, idx_to_char, vocab_size

if __name__ == "__main__":
    (corpus_indices, char_to_idx, idx_to_char, vocab_size) = load_data_jay_lyrics()
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
    model = RNNModel(model_name, vocab_size, embed_dim, hidden_dim, num_layers, dropout_rate)
"""
"""
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
"""