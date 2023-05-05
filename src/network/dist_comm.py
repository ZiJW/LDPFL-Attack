import os
import torch
import torch.distributed as dist

from base_comm import base_comm
import param

# -------------------------- torch.distributed based ---------------------------------------

class dist_comm(base_comm):
    def __init__(self, id, size, backend=param.DIST_BACKEND):
        super().__init__(id, size)
        
        """ Initialize the distributed environment. """
        os.environ['MASTER_ADDR'] = param.IP_ADDRESS
        os.environ['MASTER_PORT'] = param.IP_PORT
        dist.init_process_group(backend, rank=id, world_size=size)

    def initialize(self):
        pass

    def send(self, dst, msg):
        dist.send(tensor=msg, dst=dst)

    def recv(self, src, shape=None, type=torch.float32):
        assert shape != None, ("Need the shape of Tensor in dist_comm!")
        msg = torch.zeros(shape, dtype=type)
        dist.recv(tensor=msg, src=src)
        return msg