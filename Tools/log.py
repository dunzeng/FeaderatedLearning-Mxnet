
class log:
    # 维护系统运行日志
    def __init__(self, log_file_root):
        # 维护系统信息
        self.__log = log_file_root

    def log_info(self, info):
        # 将info信息存入log文件
        path = self.__log + "\\log"
        file_handle = open(path,"a")
        file_handle.write(str(info))
        file_handle.close()

    def new_log_file(self, file_name, info):
        # 将info信息输入file_name文件中
        path = self.__log + "\\" + str(file_name)
        file_handle = open(path,"w")
        file_handle.write(str(info)+"\n")
        file_handle.close()
        