import threading
import numpy as np
from Common.Grpc.fl_grpc_pb2_grpc import FL_GrpcServicer, add_FL_GrpcServicer_to_server
from concurrent import futures
import grpc
import time

con = threading.Condition()
num = 0

data_ori = {}
data_upd = []


class FlGrpcServer(FL_GrpcServicer):
    def __init__(self, config):
        super(FlGrpcServer, self).__init__()
        self.config = config

    def process(self, dict_data, handler):
        global num, data_ori, data_upd

        data_ori.update(dict_data)

        con.acquire()
        num += 1
        if num < self.config.num_workers:
            con.wait()
        else:
            rst = [data_ori[k] for k in sorted(data_ori.keys())]
            rst = np.array(rst).flatten()
            data_upd = handler(data_in=rst)
            data_ori = {}
            num = 0
            con.notifyAll()

        con.release()

        return data_upd

    def start(self):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=self.config.grpc_options)
        add_FL_GrpcServicer_to_server(self, server)

        target = self.address + ":" + str(self.port)
        #target = '192.168.126.12:5000'
        server.add_insecure_port(target)
        server.start()

        try:
            while True:
                time.sleep(60 * 60 * 24)
        except KeyboardInterrupt:
            server.stop(0)
