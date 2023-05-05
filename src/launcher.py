import param
from util import set_random_seed

from server import LDPFL_server
from client import LDPFL_client

import torch.multiprocessing as mp

def client_run(*args):
    set_random_seed(param.SEED)
    A = LDPFL_client(*args)
    A.evaluate()

def server_run(*args):
    set_random_seed(param.SEED)
    A = LDPFL_server(*args)
    A.evaluate()

if __name__ == "__main__":
    mp.set_start_method("spawn")

    p = mp.Process(target=server_run, args=())
    processes = [p]

    for idx in range(1, param.N_NODES):
        p = mp.Process(target=client_run, args=(idx,))
        processes.append(p)

    for p in processes:
        p.start()

    for p in processes:
        p.join()