# -*- coding: utf-8 -*-

import time
import torch
import torchvision
from torch import nn, optim
import numpy as np
import pdb
from Common.Utils.options import args_parser
import random
from torch.utils.data import DataLoader, random_split

def load_data_fashion_mnist(args, root='./Data/FashionMNIST'):
    """Download the fashion mnist dataset and then load into memory."""
    transforms = torchvision.transforms
    fmnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                        ]))
    fmnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
    num_items = len(fmnist_train)// args.num_workers # args.num_workers the # of clients  #split #1-10 client use same dataset
    dict_users, all_idxs = {}, [i for i in range(len(fmnist_train))]
    for i in range(args.num_workers):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        dict_users[i] = list(dict_users[i])
    
    distributed_fmnist_train = [list() for _ in range(args.num_workers)]
    for i in range(args.num_workers):
        for j in range(len(dict_users[i])):
            distributed_fmnist_train[i].append(fmnist_train[dict_users[i][j]])

    for i in range(args.num_workers):
        torch.save(distributed_fmnist_train[i], root+'/'+'fmnist_train_'+str(i)+'_.pt')

    #train_iter = torch.utils.data.DataLoader(distributed_fmnist_train[0], batch_size=batch_size, shuffle=True, num_workers=num_workers)
    #test_iter = torch.utils.data.DataLoader(fmnist_test, batch_size=batch_size, shuffle=True, num_workers=num_workers)



def load_data_mnist(args, root='./Data/MNIST'):
    """Download the fashion mnist dataset and then load into memory."""
    transforms = torchvision.transforms
    mnist_train = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                        ]))
    mnist_test = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
    server_data = []
    server_id = []
    label = np.ones(10) * 10
    for i in range(len(mnist_train)):
        if label[mnist_train[i][1]] > 0:
            server_data.append(mnist_train[i])
            label[mnist_train[i][1]] -= 1
            server_id.append(i)
        if np.all(label == 0):
            break
    num_items = (len(mnist_train)-100)// args.num_workers
    dict_users, all_idxs = {}, [i for i in range(len(mnist_train))]
    all_idxs = list(set(all_idxs) - set(server_id))
    for i in range(args.num_workers):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        dict_users[i] = list(dict_users[i])
    
    distributed_mnist_train = [list() for _ in range(args.num_workers)]
    for i in range(args.num_workers):
        for j in range(len(dict_users[i])):
            distributed_mnist_train[i].append(mnist_train[dict_users[i][j]])
    
    for i in range(args.num_workers):
        torch.save(distributed_mnist_train[i], root+'/'+'mnist_train_'+str(i)+'_.pt')
    torch.save(server_data, root + '/' + 'server_data.pt')
    #train_iter = torch.utils.data.DataLoader(distributed_mnist_train[0], batch_size=batch_size, shuffle=True, num_workers=0)
    #test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    #return train_iter, test_iter

def load_data_cifar10(args, root='./Data/CIFAR10'):
    transforms = torchvision.transforms
    trans_aug = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
  
    cifar10_train = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=trans_aug)
    cifar10_test = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=trans_aug)
    server_data = []
    server_id = []
    label = np.ones(10) * 10
    for i in range(len(cifar10_train)):
        if label[cifar10_train[i][1]] > 0:
            server_data.append(cifar10_train[i])
            label[cifar10_train[i][1]] -= 1
            server_id.append(i)
        if np.all(label == 0):
            break
    num_items = (len(cifar10_train)-100)// args.num_workers
    dict_users, all_idxs = {}, [i for i in range(len(cifar10_train))]
    all_idxs = list(set(all_idxs) - set(server_id))

    dict_users, all_idxs = {}, [i for i in range(len(cifar10_train))]
    for i in range(args.num_workers):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        dict_users[i] = list(dict_users[i])
    
    distributed_cifar10_train = [list() for _ in range(args.num_workers)]
    for i in range(args.num_workers):
        for j in range(len(dict_users[i])):
            distributed_cifar10_train[i].append(cifar10_train[dict_users[i][j]])    
    for i in range(args.num_workers):
        torch.save(distributed_cifar10_train[i], root+'/'+'cifar10_train_'+str(i)+'_.pt')
    torch.save(server_data, root + '/' + 'server_data.pt')
    #train_iter = torch.utils.data.DataLoader(distributed_cifar10_train[0], batch_size=batch_size, shuffle=True, num_workers=num_workers)
    #test_iter = torch.utils.data.DataLoader(cifar10_test, batch_size=batch_size, shuffle=True, num_workers=num_workers)

def load_data_tud_split(args, dataset):
    n = int(len(dataset)*0.8)
    train_dataset = dataset[:n]
    test_dataset = dataset[n:]
    print("Total training dataset: %d"%len(train_dataset))
    print("Total testing dataset: %d"%len(test_dataset))

    client_train_num = int(n/args.num_workers)
    length = [client_train_num]*(args.num_workers-1)
    length.append(n-(args.num_workers-1)*client_train_num)
    length.append(len(test_dataset))
    partition_data = random_split(dataset, length) # split training data and test data

    return partition_data


if __name__ == '__main__':
    args = args_parser()
    load_data_mnist(args, root='/home/dy/Awesome_FL/Data')
