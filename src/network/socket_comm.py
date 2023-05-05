import socket
import pickle
from time import sleep

import param
from base_comm import base_comm

# -------------------------- Socket & pickle based ------------------------------------------

class socket_comm(base_comm):
    def __init__(self, id: int, size: int) -> None:
        super().__init__(id, size)

        self.sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if id == 0:
            self.sk.bind((param.IP_ADDRESS, int(param.IP_PORT)))
            self.sks = [None] * self.size
            
    def initialize(self):
        if self.id == 0:
            self.sk.listen(5)
            for idx in range(self.size - 1):
                conn, addr = self.sk.accept()
                real_id = pickle.loads(conn.recv(param.BUFFER_SIZE))
                print("Server: Client {} connected.".format(real_id))
                self.sks[real_id] = conn
        else:
            while True:
                try:
                    self.sk.connect((param.IP_ADDRESS, int(param.IP_PORT)))
                    self.sk.send(pickle.dumps(self.id))
                    print("Client {}: Build connnection.".format(self.id))
                    break
                except:
                    print("Client {}: waiting for connnection ...".format(self.id))
                    sleep(2)
    
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
            pkg = pickle.dumps({"id": self.id, "msg": msg})
            sz = len(pkg)

            head = pickle.dumps({"id": self.id, "type": "long", "length": sz})
            assert sk.send(head) == len(head)
            assert self.recv(dst) == "READY"
            sk.sendall(pkg)

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
            msg = b""
            length = head["length"]
            self.send(src, "READY")
            while length > 0:
                data = sk.recv(param.BUFFER_SIZE)
                length -= len(data)
                msg += data
            msg = pickle.loads(msg)
        else:
            raise ValueError("Invalid lentype: {}".format(head["type"]))
        
        if restype == "raw":
            return msg
        elif restype == "check":
            assert msg["id"] == src
            return msg["msg"]
        else:
            raise ValueError("Invalid restype: {}".format(restype))