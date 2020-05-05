# FIFL：Fast Implementation of Federated Learning

架构简述：FIFL分为三层架构：网络层、逻辑层、计算层，网络层定义Server与Client端通信协议，逻辑层定义对模型或梯度处理算法，计算层定义模型结构，模型训练算法。



## 网络层

> Class: FedServer.server():

网络通信类，此类基于Socket类中TCP网络通信协议为使用者提供了基础的网络通信方法：文件接收与发送、类的接收与发送。

```python
def __init__(self, model, input_shape, init_model_randomly, init_model_path=""):
        # model: mxnet.gluon.nn.Sequential()类，传入用户自定义模型；
        # input_shape: 单个训练数据的形状元组，用于生成初始化模型；
        # init_model_randomly: True真则随机生成初始化训练模型，False则从init_model_path路径指向的模型文件读入模型参数；
        # decay_factor: 学习率衰减参数
```

### Methods

|      |      |
| ---- | ---- |
|      |      |
|      |      |
|      |      |
|      |      |
|      |      |
|      |      |
|      |      |
|      |      |
|      |      |

