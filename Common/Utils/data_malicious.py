import torch
import torchvision
import pdb

def poison_data_mnist(path=None):
    for id in range(10):
        data = torch.load(path+'/'+'mnist_train_'+str(id)+'_.pt')
        posion_data = []
        for i in range(len(data)):
            posion_data.append((data[i][0], 9 - data[i][1]))
        torch.save(posion_data, path +'/' + 'mnist_train_posioned' + str(id) +'_.pt')
    print("mnist posioned end")


def poison_data_cifar10(path=None):
    for id in range(10):
        data = torch.load(path+'/'+'cifar10_train_'+str(id)+'_.pt')
        posion_data = []
        for i in range(len(data)):
            posion_data.append((data[i][0], 9 - data[i][1]))
        torch.save(posion_data, path +'/' + 'cifar10_train_posioned' + str(id) +'_.pt')
    print("posioned end")

def poison_data_fmnist(path=None):
    for id in range(10):
        data = torch.load(path+'/'+'fmnist_train_'+str(id)+'_.pt')
        posion_data = []
        for i in range(len(data)):
            posion_data.append((data[i][0], 9 - data[i][1]))
        torch.save(posion_data, path +'/' + 'fmnist_train_posioned' + str(id) +'_.pt')
    print("fmnist posioned end")



if __name__ == '__main__':
    path = ''
    poison_data_fmnist(path=path)