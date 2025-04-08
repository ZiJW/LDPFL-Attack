import param
from util import set_random_seed
from server.LDP_FL import LDPFL_server
from client.LDP_FL import LDPFL_client
from server.FedSel import FedSel_server
from client.FedSel import FedSel_client
from server.DP_SGD import DPSGD_server
from client.DP_SGD import DPSGD_client
from server.Test_FL import Test_server
from client.Test_FL import Test_client
from server.FedAvg import FedAvg_server
from client.FedAvg import FedAvg_client
import torch.multiprocessing as mp
def client_run(*args):
    set_random_seed(param.SEED)
    clients = {
        "DPSGD": DPSGD_client,
        "LDPFL": LDPFL_client,
        "FedSel": FedSel_client,
        "Test": Test_client,
        "FedAvg": FedAvg_client
    }

    A = clients.get(param.FL_RULE, lambda *args: None)(*args)
    A.evaluate() 

def server_run(*args):
    set_random_seed(param.SEED)
    servers = {
        "DPSGD": DPSGD_server,
        "LDPFL": LDPFL_server,
        "FedSel": FedSel_server,
        "Test": Test_server,
        "FedAvg": FedAvg_server
    }

    A= servers.get(param.FL_RULE, lambda *args: None)(*args)
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
        