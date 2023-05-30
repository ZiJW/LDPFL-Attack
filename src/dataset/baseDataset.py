from torch.utils.data import Dataset
from torchvision import datasets, transforms
from adultDataset import Adult_dataset

class base_dataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

def get_dataset(dataset: str):
    if dataset == "MNIST":
        train_dataset = datasets.MNIST('./', train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                ]))

        test_dataset = datasets.MNIST('./', train=False, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                ]))
    elif dataset == "adult":
        train_dataset = Adult_dataset('./', train=True)
        test_dataset = Adult_dataset('./', train=False)
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))
    
    return train_dataset, test_dataset