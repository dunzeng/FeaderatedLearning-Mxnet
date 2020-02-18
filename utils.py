import mxnet as mx
import mxnet.ndarray as nd
import pickle
from tqdm import tqdm
import numpy as np

def network_layers_filter(network):
    """
    输入：神经网络
    功能：提取神经网络内部结构和参数
    输出：神经网络结构字典
    
    当处于卷积神经网络卷积层时会报错，待修改。
    """
    params = {}
    depth = 0
    while True:
        try:
            layer_weight = "layer"+str(depth)+"_weight"
            params[layer_weight] = network[depth].weight.data()
            layer_bias = "layer"+str(depth)+"_bias"
            params[layer_bias] = network[depth].bias.data()
            print(('Successfully parse layer %d')%depth)
            depth+=1
        except:
            print(('Ended in layer%d')%depth)
            break
    return params,depth


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
def direct_gradient(network_old,network_new,learning_rate):
    gradient = {}
    depth = 0
    while True:
        try:
            gradient_weight = network_old[depth].weight.data()[:]-network_new[depth].weight.data()[:]
            gradient['weight'+str(depth)] = gradient_weight/learning_rate
            gradient_bias = network_old[depth].bias.data()[:]-network_new[depth].bias.data()[:]
            gradient['bias'+str(depth)] = gradient_bias/learning_rate
        except:
            break
        depth+=1
    return gradient

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
    