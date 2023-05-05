import network.socket_comm
import network.dist_comm
import network.fake_comm

dist_comm = network.dist_comm.dist_comm
socket_comm = network.socket_comm.socket_comm
fake_comm = network.fake_comm.fake_socket_comm

def load_comm(type: str, id: int, size: int):
    if type == "dist":
        comm = dist_comm(id, size)
    elif type == "socket":
        comm = socket_comm(id, size)
    elif type == "fake_socket":
        comm = fake_comm(id, size)
    else:
        raise ValueError("Invalid Communication type: {}".format(comm))  
    return comm