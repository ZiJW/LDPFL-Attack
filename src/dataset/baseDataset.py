from torchvision import datasets, transforms
from torch.utils.data import Dataset

DEFAULT_DATASET_PATH = "./"

def get_dataset(dataset: str):
    if dataset == "MNIST":
        train_dataset = datasets.MNIST(DEFAULT_DATASET_PATH, train=True, download=True, transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                ]))
        test_dataset = datasets.MNIST(DEFAULT_DATASET_PATH, train=False, download=True, transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                ]))
    elif dataset == "CIFAR10":
        train_dataset = datasets.CIFAR10(DEFAULT_DATASET_PATH, train=True, download=True)
        test_dataset = datasets.CIFAR10(DEFAULT_DATASET_PATH, train=False, download=True)
    elif dataset == "CIFAR100":
        train_dataset = datasets.CIFAR100(DEFAULT_DATASET_PATH, train=True, download=True)
        test_dataset = datasets.CIFAR100(DEFAULT_DATASET_PATH, train=False, download=True)
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))
    
    return train_dataset, test_dataset


transform_MNIST_train = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(), 
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.1307,), (0.3081,))
                    ])

transform_MNIST_test = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.1307,), (0.3081,))
                    ])

transform_CIFAR10_train = transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),                                
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                    ])

transform_CIFAR10_test = transforms.Compose([                               
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                    ])

transform_CIFAR100_train = transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),                                
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                               std=[0.267, 0.256, 0.276])
                    ])

transform_CIFAR100_test = transforms.Compose([                              
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                               std=[0.267, 0.256, 0.276])
                    ])

class base_dataset(Dataset):
    def __init__(self, x, y, transform=None):
        super().__init__()
        self.x = x
        self.y = y
        self.transform = transform
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        if self.transform: 
            return self.transform(self.x[index]), self.y[index]
        else:
            return self.x[index], self.y[index]
