import os,sys

# 主目录路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
print(BASE_DIR)


# 网络层参数

# server端监听端口
IP_PORT = ("localhost",8080)

# 逻辑层参数
# 模型输入数据形状
SHAPE = (1,28,28)
# 模型保存路径
ModelSavePath = BASE_DIR + "\\demo\\current_model.params"
# 模型初始化
RandomInit = False
# 初始化模型文件
InitModelFile = BASE_DIR + "\\demo\\init_model.params"

# Client参数
# 训练参数 
BatchSize = 100
LR = 0.02
Epoch = 5
# Client端返回信息标志  Weight：Client返回模型  Gradient: Client返回梯度  Defined： 自定义
Mode = "Weight"
