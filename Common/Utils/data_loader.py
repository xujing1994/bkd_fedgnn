import torch
import torchvision
import pdb
from torch_geometric.data import DataLoader

def load_data_mnist(id, batch=None, path=None):
    data = torch.load(path+'/'+'mnist_train_'+str(id)+'_.pt')
    train_iter = torch.utils.data.DataLoader(data, batch_size=batch, shuffle=True, num_workers=0)
    if (id == 0):
        transforms = torchvision.transforms
        test = torchvision.datasets.MNIST(root=path, train=False, download=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
        test_iter = torch.utils.data.DataLoader(test, batch_size=batch, shuffle=True, num_workers=0)
        return train_iter, test_iter

    return train_iter

    # data = torch.load(path+'/'+'mnist_train_'+str(id)+'_.pt')
    # train_iter = torch.utils.data.DataLoader(data, batch_size=batch, shuffle=True, num_workers=0)
    #
    # transforms = torchvision.transforms
    # test = torchvision.datasets.MNIST(root=path, train=False, download=False, transform=transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307,), (0.3081,))
    # ]))
    # test_iter = torch.utils.data.DataLoader(test, batch_size=batch, shuffle=True, num_workers=0)
    # return train_iter, test_iter

def load_data_posioned_mnist(id, batch=None, path=None):
    data = torch.load(path+'/'+'mnist_train_posioned'+str(id)+'_.pt')
    train_iter = torch.utils.data.DataLoader(data, batch_size=batch, shuffle=True, num_workers=0)
    if (id == 0) or (id == 9):
        transforms = torchvision.transforms
        test = torchvision.datasets.MNIST(root=path, train=False, download=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
        test_iter = torch.utils.data.DataLoader(test, batch_size=batch, shuffle=True, num_workers=0)
        return train_iter, test_iter

    return train_iter

def load_data_usps(id, batch=None, path=None):
    data = torch.load(path+'/'+'usps_train_'+str(id)+'_.pt')
    train_iter = torch.utils.data.DataLoader(data, batch_size=batch, shuffle=True, num_workers=0)
    if id == 0:
        transforms = torchvision.transforms
        test = torchvision.datasets.USPS(root=path, train=False, download=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
        test_iter = torch.utils.data.DataLoader(test, batch_size=batch, shuffle=True, num_workers=0)
        return train_iter, test_iter

    return train_iter

def load_data_fmnist(id, batch=None, path=None):
    data = torch.load(path+'/'+'fmnist_train_'+str(id)+'_.pt')
    train_iter = torch.utils.data.DataLoader(data, batch_size=batch, shuffle=True, num_workers=0)
    if id == 0:
        transforms = torchvision.transforms
        test = torchvision.datasets.FashionMNIST(root=path, train=False, download=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
        test_iter = torch.utils.data.DataLoader(test, batch_size=batch, shuffle=True, num_workers=0)
        return train_iter, test_iter

    return train_iter

def load_data_cifar10(id, batch=None, path=None):
    data = torch.load(path+'/'+'cifar10_train_'+str(id)+'_.pt')
    train_iter = torch.utils.data.DataLoader(data, batch_size=batch, shuffle=True, num_workers=0)
    if id == 0:
        transforms = torchvision.transforms
        trans_aug = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        test = torchvision.datasets.CIFAR10(root=path, train=False, download=False, transform=trans_aug)
        test_iter = torch.utils.data.DataLoader(test, batch_size=batch, shuffle=True, num_workers=0)
        return train_iter, test_iter

    return train_iter

def load_data_tud(id, partition_data, test_dataset, batch=None):
    train_loader = DataLoader(partition_data[id], batch_size=batch, shuffle=True)
    if id == 0:
        test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=True)
        print("client %d training dataset: %d"%(id, len(partition_data[id])))
        return train_loader, test_loader
    print("client %d training dataset: %d"%(id, len(partition_data[id])))
    return train_loader
    
