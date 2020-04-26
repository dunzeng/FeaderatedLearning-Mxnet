import sys
path_base = "E:\\PythonProjects\\Mxnet_FederatedLearning"
sys.path.append(path_base)

from Tools.log import log
import time
import json

class server_log(log):
    def __init__(self,log_file_root):
        log.__init__(self,log_file_root)
        self.accuracy = []
        self.communication_round = 0
        with open(path_base+"\\Fed_Server\\client_train_param.json",'r') as f:
            json_data = json.load(f)
        self.write_data = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + "\n" + str(json_data) + "\n"
    
    def record_acc(self, accuracy):
        self.accuracy.append(accuracy)

    def add_cummu_round(self):
        self.communication_round += 1

    def add_data(self,data):
        self.write_data += data + "\n"
        
    def record_to_file(self):
        Tim = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.write_data += str(self.communication_round) + "\n"
        self.write_data += str(self.accuracy) + "\n"
        file_name ="Train_Information"+Tim
        print("Server Log: 日志文件写入", file_name)
        self.new_log_file(file_name,self.write_data)
        