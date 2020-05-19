# FIFL：Fast Implementation of Federated Learning

架构简述：FIFL分为三层架构：网络层、逻辑层、计算层，网络层定义Server与Client端通信协议，逻辑层定义对模型或梯度处理算法，计算层定义模型结构，模型训练算法。

## Server端

> FedServer(serve) 

网络通信类，此类基于Socket类中TCP网络通信协议为使用者提供了基础的网络通信方法：文件接收与发送、类的接收与发送

### Functions

| 方法名                         | 解释                   |
| ------------------------------ | ---------------------- |
| listen(self)                   | 监听端口               |
| multithread_listen(self)       | 监听端口多线程处理请求 |
| message_handler(self, connect) | 处理Client请求         |



> ServerHandler

逻辑层处理类，此类负责处理上层传递的请求和下层上传的数据信息。

### Functions

| 方法名                                    | 解释                               |
| ----------------------------------------- | ---------------------------------- |
| save_net2file(self, save_path)            | 将模型储存到指定路径               |
| validate_model(self, val_data, batch=100) | 使用val_data验证当前模型           |
| weight_process(self, weight_data)         | 权值更新处理                       |
| gradient_process(self, gradient_dict)     | 梯度更新处理                       |
| defined_process(self, data)               | 自定义逻辑处理算法                 |
| process_data_from_client(self, data)      | 根据Mode的不同，调用不同的相应函数 |



### 配置文件 server_config.py

| 变量名               | 解释                         |
| -------------------- | ---------------------------- |
| IP_PORT              | 监听本地端口数据             |
| SHAPE                | 训练数据形状，用于初始化模型 |
| ModelSavePath        | 模型本地保存路径             |
| RandomInit           | T/F，是否随机初始化模型      |
| InitModelFile        | 本地模型初始化文件路径       |
| **Client端控制参数** |                              |
| BatchSize            | 批量大小                     |
| LR                   | 学习率                       |
| Epoch                | 训练轮次                     |
| Mode                 | 训练模式                     |



## Client端

> FedClient(serve)

### Functions

| 方法名 | 解释 |
| ------ | ---- |
|        |      |
|        |      |
|        |      |

> FedClientHandler

### Functions

| 方法名                                              |                                                              |
| --------------------------------------------------- | ------------------------------------------------------------ |
| train_data_loader(self, batch_size=100)             | 训练数据加载函数，返回NDArrayIter                            |
| get_model(self)                                     | 获得本地模型拷贝                                             |
| get_gradient(self)                                  | 获得梯度信息拷贝                                             |
| load_model(self, model_path)                        | 加载指定路径的模型                                           |
| local_train(self, epoch, learning_rate, batch_size) | 本地训练, 会自动调用train_data_loader函数，用户需继承该类并重写数据加载函数。 |



### 配置文件Client_config.py

| 变量名                          | 解释                         |
| ------------------------------- | ---------------------------- |
| IP_PORT                         | server端IP地址，用于请求服务 |
| Data_Shape                      | 数据形状，用于初始化模型     |
| ModelPath                       | 模型保存路径                 |
| Train_Data_Path\Train_Data_Path | 训练数据\标签路径            |
| Val_Data_Path\Val_Label_Path    | 验证数据\标签路径            |



## 网络工具类 serve

| 方法名                                | 解释                                                         |
| ------------------------------------- | ------------------------------------------------------------ |
| send_class(self, connect, class_data) | 采用pickle序列化类实例，将信息通过connect发送，connect为socket网络连接成功后的网络类。 |
| recv_class(self, connect)             | 接收来自connect的信息，并通过pickle还原。                    |
| send_code(self, addr, msg_code)       | 发送请求，由Client端发起，建立并保留网络连接。               |
| recv_code(self, connect)              | 接受网络请求。                                               |
| send_file(self, connect, file_path)   | 将file_path路径下的文件以二进制发送。                        |
| recv_file(self, save_path)            | 接收文件，并存入save_path路径下。                            |

