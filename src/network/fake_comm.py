import os
import pickle
from time import sleep

import param
from socket_comm import socket_comm

class fake_socket_comm(socket_comm):
    """
        Use socket to send short message.
        For big file, save them in /tmp/ to simulate communication. 
        ** ONLY ON LOCALHOST ** 
    """
    def __init__(self, id: int, size: int) -> None:
        assert param.IP_ADDRESS == "127.0.0.1", "fake socket is only available on localhost!"
        super().__init__(id, size)
        self.path = "/tmp/fake_"
        
    def send(self, dst: int, msg):
        assert self.id == 0 or dst == 0
        if self.id == 0:
            sk = self.sks[dst]
        else:
            sk = self.sk

        pkg = pickle.dumps({"id": self.id, "type": "short", "msg": msg})
        sz = len(pkg)
        if 2 * sz <= param.BUFFER_SIZE:
            # short
            assert sk.send(pkg) == sz
        else:
            # long
            name = self.path + "{}->{}.pkl".format(self.id, dst)
            with open(name, "wb") as F:
                pickle.dump({"id": self.id, "msg": msg}, F)
            head = pickle.dumps({"id": self.id, "type": "long", "length": sz})
            assert sk.send(head) == len(head)
    
    def recv(self, src: int, restype: str = "check"):
        if self.id == 0:
            sk = self.sks[src]
        else:
            assert src == 0, "Client {} can only recv from Server".format(self.id)
            sk = self.sk

        head = pickle.loads(sk.recv(param.BUFFER_SIZE))
        if head["type"] == "short":
            msg = head
            msg.pop("type")
        elif head["type"] == "long":
            name = self.path + "{}->{}.pkl".format(src, self.id)
            while True:
                if os.path.exists(name):
                    with open(name, "rb") as F:
                        msg = pickle.load(F)
                    os.remove(name)
                    break
                sleep(0.1)
        else:
            raise ValueError("Invalid lentype: {}".format(head["type"]))
        
        if restype == "raw":
            return msg
        elif restype == "check":
            assert msg["id"] == src
            return msg["msg"]
        else:
            raise ValueError("Invalid restype: {}".format(restype))