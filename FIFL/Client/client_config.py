import os,sys

# 主目录路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
print(BASE_DIR)

# Server ip 地址
IP_PORT = ("localhost",8080)

# 数据形状
Data_Shape = (1,28,28)

# 模型保存路径
ModelPath = BASE_DIR + "\\demo\\client_model.params"

# 训练数据路径
Train_Data_Path = BASE_DIR + "\\demo\\data\\train_data0.npy"
Train_Label_Path = BASE_DIR + "\\demo\\data\\train_label0.npy"

# 验证数据路径
Val_Data_Path = BASE_DIR + "\\demo\\data\\"
Val_Label_Path = BASE_DIR + "\\demo\\data\\"

# SGD训练数据
Batch_size = 100
LR = 0.01
Epoch = 10