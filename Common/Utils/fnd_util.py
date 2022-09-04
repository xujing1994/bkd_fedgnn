import networkx as nx
import random
import copy
from networkx.linalg.laplacianmatrix import _transition_matrix
import torch
import pickle
import os.path as osp

#from torch_geometric.datasets import TUDataset
#from torch_geometric.utils import degree, convert
#import torch_geometric.transforms as T
import os
#import torch_geometric
import numpy as np
import copy
#from torch_geometric.datasets import Planetoid
#from torch_geometric.nn.conv.gcn_conv import gcn_norm
import dgl
import glob
import matplotlib.pyplot as plt

class DGLFormDataset(torch.utils.data.Dataset):
    """
        DGLFormDataset wrapping graph list and label list as per pytorch Dataset.
        *lists (list): lists of 'graphs' and 'labels' with same len().
    """
    def __init__(self, *lists):
        assert all(len(lists[0]) == len(li) for li in lists)
        self.lists = lists
        self.graph_lists = lists[0]  # graphs
        self.graph_labels = lists[1] # labels

    def __getitem__(self, index):
        return tuple(li[index] for li in self.lists)

    def __len__(self):
        return len(self.lists[0])

def format_dataset(dataset):  
    """
        Utility function to recover data,
        INTO-> dgl/pytorch compatible format 
    """
    graphs = [data[0] for data in dataset]
    labels = [data[1] for data in dataset]

    for graph in graphs:
        #graph.ndata['feat'] = torch.FloatTensor(graph.ndata['feat'])
        #graph.ndata['feat'] = graph.ndata['feat'].float() # dgl 4.0
        # adding edge features for Residual Gated ConvNet, if not there
        if 'feat' not in graph.edata.keys():
            edge_feat_dim = graph.ndata['feat'].shape[1] # dim same as node feature dim
            graph.edata['feat'] = torch.ones(graph.number_of_edges(), edge_feat_dim)

    return DGLFormDataset(graphs, labels)

def transform_dataset(trainset, testset, avg_nodes, args):
    train_untarget_idx = []
    for i in range(len(trainset)):
        if trainset[i][1].item() != args.target_label:
            train_untarget_idx.append(i)
    train_untarget_graphs = [copy.deepcopy(graph) for graph in trainset if graph[1].item() != args.target_label]
    #train_untarget_graphs = [graph for graph in trainset if graph[1].item() != args.target_label]
    #train_target_graphs = [copy.deepcopy(graph) for graph in trainset if graph[1].item() == args.target_label]
    tmp_graphs = []
    tmp_idx = []
    #train_clean_graphs = train_graphs[int(poisoning_intensity*len(train_graphs)):]

    #delete_graphs = []
    #train_trigger_graphs_final = []
    #num_nodes_min = min([train_trigger_graphs[i].g.number_of_nodes() for i in range(len(train_trigger_graphs))])
    num_trigger_nodes = int(avg_nodes * args.frac_of_avg)
    #num_trigger_nodes = 50
    for idx, graph in enumerate(train_untarget_graphs):
        if graph[0].num_nodes() > num_trigger_nodes:
            tmp_graphs.append(graph)
            tmp_idx.append(train_untarget_idx[idx])
    n_trigger_graphs = int(args.poisoning_intensity*len(trainset))
    final_idx = []
    if n_trigger_graphs <= len(tmp_graphs):
        train_trigger_graphs = tmp_graphs[:n_trigger_graphs]
        final_idx = tmp_idx[:n_trigger_graphs]
    else:
        train_trigger_graphs = tmp_graphs
        final_idx = tmp_idx
    #train_trigger_graphs_final.remove(train_trigger_graphs[0])

    G_trigger = nx.erdos_renyi_graph(num_trigger_nodes, args.density, directed=True)
    #G_trigger = dgl.DGLGraph(nx.erdos_renyi_graph(num_trigger_nodes, 0.3), directed=False)
    trigger_list = []
    for data in train_trigger_graphs:
        #trigger_num = random.sample(range(train_trigger_graphs[i][0].num_nodes()), num_trigger_nodes)
        trigger_num = random.sample(data[0].nodes().tolist(), num_trigger_nodes)
        trigger_list.append(trigger_num)

    for  i, data in enumerate(train_trigger_graphs):
        for j in range(len(trigger_list[i])-1):
            for k in range(j+1, len(trigger_list[i])):
                if data[0].has_edges_between(trigger_list[i][j], trigger_list[i][k]) and G_trigger.has_edge(j, k) is False:
                    ids = data[0].edge_ids(torch.tensor([trigger_list[i][j]]), torch.tensor([trigger_list[i][k]]))
                    data[0].remove_edges(ids)
                if data[0].has_edges_between(trigger_list[i][j], trigger_list[i][k]) is False and G_trigger.has_edge(j, k):
                    data[0].add_edges(torch.tensor([trigger_list[i][j]]), torch.tensor([trigger_list[i][k]]))
                if data[0].has_edges_between(trigger_list[i][k], trigger_list[i][j]) and G_trigger.has_edge(k, j) is False:
                    ids = data[0].edge_ids(torch.tensor([trigger_list[i][k]]), torch.tensor([trigger_list[i][j]]))
                    data[0].remove_edges(ids)
                if data[0].has_edges_between(trigger_list[i][k], trigger_list[i][j]) is False and G_trigger.has_edge(k, j):
                    data[0].add_edges(torch.tensor([trigger_list[i][k]]), torch.tensor([trigger_list[i][j]]))
    
    ## rebuild data with target label
    graphs = [data[0] for data in train_trigger_graphs]
    labels = [torch.tensor(args.target_label) for i in range(len(train_trigger_graphs))]
    train_trigger_graphs = DGLFormDataset(graphs, labels)

    test_changed_graphs = [copy.deepcopy(graph) for graph in testset if graph[1].item() != args.target_label]
    delete_test_changed_graphs = []
    test_changed_graphs_final = []
    for graph in test_changed_graphs:
        if graph[0].num_nodes() < num_trigger_nodes:
            delete_test_changed_graphs.append(graph)
    for graph in test_changed_graphs:
        if graph not in delete_test_changed_graphs:
            test_changed_graphs_final.append(graph)
    test_changed_graphs = test_changed_graphs_final
    print("num_of_test_changed_graphs is: %d"%len(test_changed_graphs_final))
    for data in test_changed_graphs:
        #print(graph[0].nodes())
        trigger_idx = random.sample(data[0].nodes().tolist(), num_trigger_nodes)
        for i in range(len(trigger_idx)-1):
            for j in range(i+1, len(trigger_idx)):
                if data[0].has_edges_between(trigger_idx[i], trigger_idx[j]) and G_trigger.has_edge(i, j) is False:
                    ids = data[0].edge_ids(torch.tensor([trigger_idx[i]]), torch.tensor([trigger_idx[j]]))
                    data[0].remove_edges(ids)
                if data[0].has_edges_between(trigger_idx[i], trigger_idx[j]) is False and G_trigger.has_edge(i, j):
                    data[0].add_edges(torch.tensor([trigger_idx[i]]), torch.tensor([trigger_idx[j]]))
                if data[0].has_edges_between(trigger_idx[j], trigger_idx[i]) and G_trigger.has_edge(j, i) is False:
                    ids = data[0].edge_ids(torch.tensor([trigger_idx[j]]), torch.tensor([trigger_idx[i]]))
                    data[0].remove_edges(ids)
                if data[0].has_edges_between(trigger_idx[j], trigger_idx[i]) is False and G_trigger.has_edge(j, i):
                    data[0].add_edges(torch.tensor([trigger_idx[j]]), torch.tensor([trigger_idx[i]]))
    graphs = [data[0] for data in test_changed_graphs]
    labels = [torch.tensor(args.target_label) for i in range(len(test_changed_graphs))]
    test_trigger_graphs = DGLFormDataset(graphs, labels)

    tmp_graphs = [trainset[idx] for idx in range(len(trainset)) if idx not in final_idx]
    graphs_1 = [data[0] for data in train_trigger_graphs]
    graphs_2 = [data[0] for data in tmp_graphs]
    labels_1 = [torch.tensor(args.target_label) for i in range(len(train_trigger_graphs))]
    labels_2 = [data[1] for data in tmp_graphs]
    graphs_1.extend(graphs_2)
    labels_1.extend(labels_2)
    bkd_train_graphs = DGLFormDataset(graphs_1, labels_1)
    return train_trigger_graphs, bkd_train_graphs, test_trigger_graphs, G_trigger, final_idx

def inject_global_trigger_test(testset, avg_nodes, args, global_trigger):
    test_changed_graphs = [copy.deepcopy(graph) for graph in testset if graph[1].item() != args.target_label]
    num_mali = len(global_trigger)
    num_trigger_nodes = int(avg_nodes * args.frac_of_avg) * num_mali
    delete_test_changed_graphs = []
    test_changed_graphs_final = []
    for graph in test_changed_graphs:
        if graph[0].num_nodes() < num_trigger_nodes:
            delete_test_changed_graphs.append(graph)
    for graph in test_changed_graphs:
        if graph not in delete_test_changed_graphs:
            test_changed_graphs_final.append(graph)
    test_changed_graphs = test_changed_graphs_final
    print("num_of_test_changed_graphs is: %d"%len(test_changed_graphs_final))
    each_trigger_nodes = int(avg_nodes * args.frac_of_avg)
    for graph in test_changed_graphs:
        #print(graph[0].nodes())
        trigger_idx = random.sample(graph[0].nodes().tolist(), num_trigger_nodes)
        for idx, trigger in enumerate(global_trigger):
            start = each_trigger_nodes * idx
            for i in range(start, start+each_trigger_nodes-1):
                for j in range(i+1, start+each_trigger_nodes):
                    if graph[0].has_edges_between(trigger_idx[i], trigger_idx[j]) and trigger.has_edge(i, j) is False:
                        ids = graph[0].edge_ids(torch.tensor([trigger_idx[i]]), torch.tensor([trigger_idx[j]]))
                        graph[0].remove_edges(ids)
                    if graph[0].has_edges_between(trigger_idx[i], trigger_idx[j]) is False and trigger.has_edge(i, j):
                        graph[0].add_edges(torch.tensor([trigger_idx[i]]), torch.tensor([trigger_idx[j]]))
                    if graph[0].has_edges_between(trigger_idx[j], trigger_idx[i]) and trigger.has_edge(j, i) is False:
                        ids = graph[0].edge_ids(torch.tensor([trigger_idx[j]]), torch.tensor([trigger_idx[i]]))
                        graph[0].remove_edges(ids)
                    if graph[0].has_edges_between(trigger_idx[j], trigger_idx[i]) is False and trigger.has_edge(j, i):
                        graph[0].add_edges(torch.tensor([trigger_idx[j]]), torch.tensor([trigger_idx[i]]))
                        
    graphs = [data[0] for data in test_changed_graphs]
    labels = [torch.tensor(args.target_label) for i in range(len(test_changed_graphs))]
    test_trigger_graphs = DGLFormDataset(graphs, labels)
    return test_trigger_graphs

def inject_global_trigger_train(trainset, avg_nodes, args, triggers):
    train_untarget_idx = []
    for i in range(len(trainset)):
        if trainset[i][1].item() != args.target_label:
            train_untarget_idx.append(i)

    train_untarget_graphs = [copy.deepcopy(graph) for graph in trainset if graph[1].item() != args.target_label]
    tmp_graphs = []
    tmp_idx = []
    num_mali = len(triggers)
    num_trigger_nodes = int(avg_nodes * args.frac_of_avg) * num_mali

    for idx, graph in enumerate(train_untarget_graphs):
        if graph[0].num_nodes() > num_trigger_nodes:
            tmp_graphs.append(graph)
            tmp_idx.append(train_untarget_idx[idx])
    n_trigger_graphs = int(args.poisoning_intensity*len(trainset))
    final_idx = []
    if n_trigger_graphs <= len(tmp_graphs):
        train_trigger_graphs = tmp_graphs[:n_trigger_graphs]
        final_idx = tmp_idx[:n_trigger_graphs]
    else:
        train_trigger_graphs = tmp_graphs
        final_idx = tmp_idx
    print("num_of_train_trigger_graphs is: %d"%len(train_trigger_graphs))
    each_trigger_nodes = int(avg_nodes * args.frac_of_avg)
    for graph in train_trigger_graphs:
        #print(graph[0].nodes())
        trigger_idx = random.sample(graph[0].nodes().tolist(), num_trigger_nodes)
        for idx, trigger in enumerate(triggers):
            start = each_trigger_nodes * idx
            for i in range(start, start+each_trigger_nodes-1):
                for j in range(i+1, start+each_trigger_nodes):
                    if graph[0].has_edges_between(trigger_idx[i], trigger_idx[j]) and trigger.has_edge(i, j) is False:
                        ids = graph[0].edge_ids(torch.tensor([trigger_idx[i]]), torch.tensor([trigger_idx[j]]))
                        graph[0].remove_edges(ids)
                        #dgl.remove_edges(graph[0], ids)
                    if graph[0].has_edges_between(trigger_idx[i], trigger_idx[j]) is False and trigger.has_edge(i, j):
                        graph[0].add_edges(torch.tensor([trigger_idx[i]]), torch.tensor([trigger_idx[j]]))
                    if graph[0].has_edges_between(trigger_idx[j], trigger_idx[i]) and trigger.has_edge(j, i) is False:
                        ids = graph[0].edge_ids(torch.tensor([trigger_idx[j]]), torch.tensor([trigger_idx[i]]))
                        graph[0].remove_edges(ids)
                        #dgl.remove_edges(graph[0], ids)
                    if graph[0].has_edges_between(trigger_idx[j], trigger_idx[i]) is False and trigger.has_edge(j, i):
                        graph[0].add_edges(torch.tensor([trigger_idx[j]]), torch.tensor([trigger_idx[i]]))
    tmp_graphs = [trainset[idx] for idx in range(len(trainset)) if idx not in final_idx]
    graphs_1 = [data[0] for data in train_trigger_graphs]
    graphs_2 = [data[0] for data in tmp_graphs]
    labels_1 = [torch.tensor(args.target_label) for i in range(len(train_trigger_graphs))]
    labels_2 = [data[1] for data in tmp_graphs]
    graphs_1.extend(graphs_2)
    labels_1.extend(labels_2)
    bkd_train_graphs = DGLFormDataset(graphs_1, labels_1)

    return bkd_train_graphs

def save_object(obj, filename):
    savedir = os.path.split(filename)[0]
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_pkl(filename):
    with open(filename, 'rb') as input:
        graphs = pickle.load(input)
    return graphs

