from abc import ABC, abstractmethod

class base_comm(ABC):
    def __init__(self, id, size) -> None:
        self.id = id
        self.size = size

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def send(self):
        pass

    @abstractmethod
    def recv(self):
        pass