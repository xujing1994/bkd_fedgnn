import torch
from torch import nn
from torch import device
import Common.config as config
import json
import os
from Common.Model.LeNet import LeNet
from Common.Model.ResNet import ResNet, BasicBlock
from Common.Model.gnn_models import GIN
#from Common.Utils.data_loader import load_data_fmnist, load_data_cifar10, load_data_mnist, load_data_tud_v2
#from Common.Utils.set_log import setup_logging
from Common.Utils.options import args_parser
from Common.Utils.gnn_util import transform_dataset, inject_global_trigger_test, save_object
import grpc
import copy
import time
from Common.Utils.evaluate import gnn_evaluate_accuracy_v2
#from Common.Utils.data_split_iid import load_data_tud_split_v2
import numpy as np
import torch.nn.functional as F
#from GNN_common.data.data import LoadData
from GNN_common.data.TUs import TUsDataset
from GNN_common.nets.TUs_graph_classification.load_net import gnn_model  # import GNNs
import random
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt 

def plot_histo_graphs(dataset, title):
    # histogram of graph sizes
    graph_sizes = []
    for graph in dataset:
        graph_sizes.append(graph[0].number_of_nodes())
        #graph_sizes.append(graph[0].number_of_edges())
    plt.figure(1)
    plt.hist(graph_sizes, bins=20)
    plt.title(title)
    plt.show()
    graph_sizes = torch.Tensor(graph_sizes)
    print('nb/min/max :',len(graph_sizes),graph_sizes.min().long().item(),graph_sizes.max().long().item())

if __name__ == '__main__':
    args = args_parser()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataset = TUsDataset(args.dataset)

    num_classes = torch.max(dataset.all.graph_labels).item() + 1
    per_class = torch.bincount(dataset.all.graph_labels)
    for i in range(per_class.shape[0]):
        print('class %d is %d'%(i, per_class[i].item()))
    plot_histo_graphs(dataset.train[0],'trainset')
    plot_histo_graphs(dataset.test[0],'testset')
    plot_histo_graphs(dataset.val[0],'valset')
