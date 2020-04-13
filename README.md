# FeaderatedLearning-Mxnet
毕业设计，联邦学习基础框架设计与实现，欢迎交流！

## 架构：

整体采用C/S架构，联邦学习分为Server端和Client端。

在测试环境中，Server端和Client端为一对多模式。

架构为三层架构：网络层、逻辑层、计算层。计算层包括模型参数维护和训练，在MXNet框架下实现。网络层和逻辑层提供了基本的数据传输和数据处理的接口，为用户提供扩展接口的算法。

### Fed_Server：

源码由Server.py和Server_data_handler.py两部分构成，server_config.json文件中为系统参数信息。

### Fed_Clinet：

源码由Client.py和Client_data_handler.py两部分构成，client_config.json文件中为系统参数信息。

## 使用方法：

暂略

