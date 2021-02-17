import torch
import torchvision
import torchvision.transforms as transforms


def get_data(batch_size, root='./data/'):
    train_dataset = torchvision.datasets.MNIST(root=root,
                                            train=True, 
                                            transform=transforms.ToTensor(),
                                            download=True)

    test_dataset = torchvision.datasets.MNIST(root=root,
                                            train=False, 
                                            transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size, 
                                            shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size, 
                                            shuffle=False)

    return train_loader, test_loader