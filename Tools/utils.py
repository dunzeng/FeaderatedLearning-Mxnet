import mxnet as mx
import mxnet.ndarray as nd
import pickle
from tqdm import tqdm
import numpy as np
import random

def network_layers_filter(network):
    """
    输入：神经网络
    功能：提取神经网络内部结构和参数
    输出：神经网络权值矩阵列表
          Todo 解析可权值网络层数列表 traverse_list
    当处于卷积神经网络卷积层时会报错，待修改。
    """
    weight = []
    bias = []
    depth = 0
    while True:
        try:
            weight.append(network[depth].weight.data()[:])
            bias.append(network[depth].bias.data()[:])
            print(('Successfully parse layer %d')%depth)
            depth+=1
        except:
            print(('Ended in layer%d')%depth)
            break
    return (weight,bias),depth


def network_grad_fileter(network):
    params = {}
    depth = 0
    while True:
        try:
            layer_weight = "layer"+str(depth)+"_weight_grad"
            params[layer_weight] = network[depth].weight.grad()
            layer_bias = "layer"+str(depth)+"_bias_grad"
            params[layer_bias] = network[depth].bias.grad()
            depth+=1
        except:
            break
    return params,depth

# 返回梯度字典 
# 待更新为梯度列表 方便梯度更新的统一化2.18
# traverse_list表示带有权值的网络参数
def direct_gradient(network_old,network_new,learning_rate,traverse_list=[]):
    gradient_w = []
    gradient_b = []
    depth = 0
    while True:
        try:
            gradient_weight = network_old[depth].weight.data()[:]-network_new[depth].weight.data()[:]
            gradient_w.append(gradient_weight/learning_rate)
            gradient_bias = network_old[depth].bias.data()[:]-network_new[depth].bias.data()[:]
            gradient_b.append(gradient_bias/learning_rate)
        except:
            break
        depth+=1
    return (gradient_w,gradient_b)

# 将data切割为大小为block_size每块的列表
def cut_bytes(data,block_size):
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

# 利用pickle模块将data按照connect连接发送
def send_class(connect, class_data):
    data = pickle.dumps(class_data)
    data_list = cut_bytes(data,1024)
    list_size = len(data_list)
    print(list_size)
    msg = str(list_size).encode('utf-8')
    connect.send(msg)
    for i in tqdm(range(list_size),desc="Sending Data"):
        connect.send(data_list[i])


# 与send_class成对使用  此端为接收方
# 1024bit为1Byte
def recv_class(connect):
    tmp_data = connect.recv(1024)
    try:
        block_size = int(tmp_data.decode())
    except:
        raise("decode error"+str(tmp_data))
    data = bytes()
    for _ in tqdm(range(block_size),desc="Recving Data"):
        data_slice = connect.recv(1024)
        data += data_slice
    class_info = pickle.loads(data)
    return class_info

# 选取从大到小排列占比为ratio的矩阵值，其余置零  传入numpy矩阵
# 计算速度过慢?
def max_parse_gradient_matrix(theta,matrix):
    sort_list = []
    mat_np = matrix.asnumpy()
    for x in np.nditer(mat_np):
        sort_list.append(abs(x))
    sort_list.sort(reverse=True)
    num = len(sort_list)
    for x in np.nditer(mat_np,op_flags=['readwrite']):
        idx = sort_list.index(abs(x))
        if idx/num >= theta :
            x[...] = 0
    return mat_np

#未测试
def max_parse_gradient_matrix_list(theta, matrix_set):
    sort_list = []
    num_of_elements = 0
    mat_np = []
    for i in range(len(matrix_set)):
        mat_np.append(matrix_set[i].asnumpy())
        for x in np.nditer(mat_np[i]):
            sort_list.append(abs(x))
    sort_list.sort(reverse=True)
    num_of_elements = len(sort_list)
    for i in range(len(matrix_set)):
        for x in np.nditer(mat_np[i],op_flags=['readwrite']):
            idx = sort_list.index(abs(x))
            if idx/num_of_elements >= theta:
                x[...] = 0
    return mat_np
    

def try_all_gpus():
    """Return all available GPUs, or [mx.cpu()] if there is no GPU."""
    ctxes = []
    try:
        for i in range(16):
            ctx = mx.gpu(i)
            _ = nd.array([0], ctx=ctx)
            ctxes.append(ctx)
    except mx.base.MXNetError:
        pass
    if not ctxes:
        ctxes = [mx.cpu()]
    return ctxes


def try_gpu():
    """If GPU is available, return mx.gpu(0); else return mx.cpu()."""
    try:
        ctx = mx.gpu()
        _ = nd.array([0], ctx=ctx)
    except mx.base.MXNetError:
        ctx = mx.cpu()
    return ctx

def data_iter_random(corpus_indices, batch_size, num_steps, ctx=None):
    #随机采样
    # 减1是因为输出的索引是相应输入的索引加1
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    # 返回从pos开始的长为num_steps的序列
    def _data(pos):
        return corpus_indices[pos: pos + num_steps]

    for i in range(epoch_size):
        # 每次读取batch_size个随机样本
        i = i * batch_size
        batch_indices = example_indices[i: i + batch_size]
        X = [_data(j * num_steps) for j in batch_indices]
        Y = [_data(j * num_steps + 1) for j in batch_indices]
        # yield语法
        # 将函数构造为一个迭代器 节约内存
        yield nd.array(X, ctx), nd.array(Y, ctx)

def data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx=None):
    #相邻采样
    corpus_indices = nd.array(corpus_indices, ctx=ctx)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[0: batch_size*batch_len].reshape((
        batch_size, batch_len))
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y