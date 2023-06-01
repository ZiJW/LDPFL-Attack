from torchvision import datasets, transforms

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
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))
    
    return train_dataset, test_dataset