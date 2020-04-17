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
    
    def record_acc(self, accuracy):
        self.accuracy.append(accuracy)

    def add_cummu_round(self):
        self.communication_round += 1

    def record_to_file(self):
        Tim = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        with open(path_base+"\\Fed_Server\\client_train_param.json",'r') as f:
            json_data = json.load(f)
        data = Tim + "\n" + str(json_data) + "\n" + str(self.communication_round) + "\n" + str(self.accuracy)
        self.new_log_file("Train_Information"+Tim,data)
        