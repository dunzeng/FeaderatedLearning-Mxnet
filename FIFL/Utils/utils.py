import mxnet as mx
import mxnet.ndarray as nd
import pickle
from tqdm import tqdm
import numpy as np
import random
import time


def cut_bytes(data,block_size):
    # 将data切割为大小为block_size每块的列表
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

def send_class(connect, class_data):
    # 利用pickle模块将data序列化，并通过connect发送
    # connect参数为socket连接对象
    data = pickle.dumps(class_data)
    data_list = cut_bytes(data,1024)
    list_size = len(data_list)
    msg = str(list_size).encode('utf-8')
    connect.send(msg)
    time.sleep(0.2)
    for i in tqdm(range(list_size),desc="Sending Data"):
        connect.send(data_list[i])

def recv_class(connect):
    # 与send_class成对使用，从socket连接中接收数据
    tmp_data = connect.recv(1024)
    try:
        block_size = int(tmp_data.decode())
    except:
        print("Error: %s"%(str(tmp_data)))
        raise Exception("Decode Error")
    data = bytes()
    for _ in tqdm(range(block_size),desc="Recving Data"):
        data_slice = connect.recv(1024)
        data += data_slice
    class_info = pickle.loads(data)
    return class_info

def try_all_gpus():
    # 返回所有可用GPUs，如果没有GPC则返回[mx.cpu()]
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
    # 返回GPU，若无则返回CPU
    try:
        ctx = mx.gpu()
        _ = nd.array([0], ctx=ctx)
    except mx.base.MXNetError:
        ctx = mx.cpu()
    return ctx