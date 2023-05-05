from base_client import Base_client
from util import log, load_dataset, load_model, load_criterion, load_optimizer
from param import DEVICE

class Single_client(Base_client):
    """
        The client in which the client is training only on its local data.
    """
    
    def __init__(self, id, size, Model, Model_param, Optimizer, Learning_rate, Criterion, Dataset, Epoch):
        super().__init__(id, size, Model, Model_param, Optimizer, Learning_rate, Criterion)

        self.train_loader, self.test_loader = load_dataset(Dataset, "fl", self.id)
        self.epoch = Epoch
    
    def train(self):
        """
            Train the model on 1 epoch.
        """
        for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(DEVICE), target.to(DEVICE)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

    def evaluate(self):
        """
            Train the model on all epoches.
            Test the model after every epoch.
        """
        for ep in range(self.epoch):
            self.train()
            self.test(ep)