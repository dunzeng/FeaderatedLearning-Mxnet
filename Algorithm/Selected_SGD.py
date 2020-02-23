import numpy as np
import copy
from mxnet import ndarray as nd

class SSGD_Server:
    # Selected SGD algorithm
    # From paper Privacy-Preserving Deep Learning
    # class for Server
    def __init__(self,network,traverse_list,theat_d,lambda_,tao):
        self.stat = []
        self.max_stat = 0
        self.theat_d = theat_d # 模型参数选择比例
        self.lambda_ = lambda_
        self.tao = tao
        self.__init_stat(network,traverse_list)

    def __init_stat(self,network,traverse_list):
        # 根据network结构初始化stat列表
        for idx in traverse_list:
            this_shape = network[idx].weight.data().shape
            self.stat.append(np.zeros(shape=this_shape))

    def get_model_selected(self,origin_net,save_path,traverse_list,threshold=0):
        # 根据stat列表
        # origin_net选择Server端模型参数
        # 并将选择好的模型保存save_path
        tmp_net = copy.deepcopy(origin_net)
        for idx in traverse_list:
            print("Selecting layer %d"%(idx))
            #numpy version
            data_np = tmp_net[idx].weight.data().asnumpy()
            # 将最近更新的权值矩阵选出
            # 置零更新频率低的参数
            data_np[self.stat[idx]<=self.max_stat-threshold] = 0
            data_nd = nd.array(data_np)
            tmp_net[idx].weight.data()[:] = data_nd
        tmp_net.save_parameters(save_path)
    
    def updata_stat(self,gradient_list):
        # 根据服务端传入的梯度信息
        # 更新本地stat参数更新计数器列表
        idx = 0
        for value in gradient_list.items():
            mat_np = value.asnumpy()
            mat_np[mat_np!=0] = 1
            self.stat[idx] = self.stat[idx] + nd.array(mat_np)
            idx+=1

    
class SSGD_Client:
    # Selected SGD algorithm
    # From paper Privacy-Preserving Deep Learning
    # class for Client
    def __init__(self,theta_u,tao_):
        # theta_u梯度选择上传比例
        self.theta_u = 0.8 # 梯度选择上传比例
        self.tao_ = 0
    
    #waiting implementation
    def max_select_weight(self,gradient_matrix):
        # 根据theta_u的值作为选择比例
        # 筛选weight比例大于theta_u的参数
        # 其余weight对应位置置0
        pass

    def get_gradient_selected(self,gradient_list):
        # 选择gradient_list参数
        # 返回selected gradient_list
        pass

    def load_selected_model(self,network,selected_model_path,traverse_list):
        # 将selected_model_path指向的模型
        # 加载至network中
        # warning：selected_model中存在0，因此不能直接通过net.load_parameters方法直接读入
        tmp_net = copy.deepcopy(network)
        tmp_net.load_parameters(selected_model_path)
        for idx in traverse_list:
            data_np = network[idx].weight.data().asnumpy()
            layer_np = network[idx].weight.data()[:].asnumpy()
            layer_np[data_np!=0] = 0
            layer_np += data_np
            layer_nd = nd.array(layer_np)
            network[idx].weight.data()[:] = layer_nd