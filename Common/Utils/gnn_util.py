import networkx as nx
import random
import copy
from networkx.linalg.laplacianmatrix import _transition_matrix
import torch
import pickle
import os.path as osp

import os
import numpy as np
import copy
import dgl
from torch.utils.data import random_split

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

def transform_dataset(trainset, testset, avg_nodes, args):
    train_untarget_idx = []
    for i in range(len(trainset)):
        if trainset[i][1].item() != args.target_label:
            train_untarget_idx.append(i)

    train_untarget_graphs = [copy.deepcopy(graph) for graph in trainset if graph[1].item() != args.target_label]
    tmp_graphs = []
    tmp_idx = []
    num_trigger_nodes = int(avg_nodes * args.frac_of_avg)
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

    G_trigger = nx.erdos_renyi_graph(num_trigger_nodes, args.density, directed=False)
    trigger_list = []
    for data in train_trigger_graphs:
        trigger_num = random.sample(data[0].nodes().tolist(), num_trigger_nodes)
        trigger_list.append(trigger_num)

    for  i, data in enumerate(train_trigger_graphs):
        for j in range(len(trigger_list[i])-1):
            for k in range(j+1, len(trigger_list[i])):
                if (data[0].has_edges_between(trigger_list[i][j], trigger_list[i][k]) or data[0].has_edges_between(trigger_list[i][k], trigger_list[i][j])) \
                    and G_trigger.has_edge(j, k) is False:
                    ids = data[0].edge_ids(torch.tensor([trigger_list[i][j], trigger_list[i][k]]), torch.tensor([trigger_list[i][k], trigger_list[i][j]]))
                    data[0].remove_edges(ids)
                elif (data[0].has_edges_between(trigger_list[i][j], trigger_list[i][k]) or data[0].has_edges_between(trigger_list[i][k], trigger_list[i][j])) is False \
                    and G_trigger.has_edge(j, k):
                    data[0].add_edges(torch.tensor([trigger_list[i][j], trigger_list[i][k]]), torch.tensor([trigger_list[i][k], trigger_list[i][j]]))
    ## rebuild data with target label
    graphs = [data[0] for data in train_trigger_graphs]
    labels = [torch.tensor([args.target_label]) for i in range(len(train_trigger_graphs))]
    train_trigger_graphs = DGLFormDataset(graphs, labels)

    test_clean_graphs = [copy.deepcopy(graph) for graph in testset]
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
    for graph in test_changed_graphs:
        trigger_idx = random.sample(graph[0].nodes().tolist(), num_trigger_nodes)
        for i in range(len(trigger_idx)-1):
            for j in range(i+1, len(trigger_idx)):
                if (graph[0].has_edges_between(trigger_idx[i], trigger_idx[j]) or graph[0].has_edges_between(trigger_idx[j], trigger_idx[i])) \
                    and G_trigger.has_edge(i, j) is False:
                    ids = graph[0].edge_ids(torch.tensor([trigger_idx[i], trigger_idx[j]]), torch.tensor([trigger_idx[j], trigger_idx[i]]))
                    graph[0].remove_edges(ids)
                elif (graph[0].has_edges_between(trigger_idx[i], trigger_idx[j]) or graph[0].has_edges_between(trigger_idx[j], trigger_idx[i])) is False \
                    and G_trigger.has_edge(i, j):
                    graph[0].add_edges(torch.tensor([trigger_idx[i], trigger_idx[j]]), torch.tensor([trigger_idx[j], trigger_idx[i]]))
    graphs = [data[0] for data in test_changed_graphs]
    labels = [torch.tensor([args.target_label]) for i in range(len(test_changed_graphs))]
    test_trigger_graphs = DGLFormDataset(graphs, labels)
    
    return train_trigger_graphs, test_trigger_graphs, G_trigger, final_idx

def transform_dataset_same_local_trigger(trainset, testset, avg_nodes, args, G_trigger):
    train_untarget_idx = []
    for i in range(len(trainset)):
        if trainset[i][1].item() != args.target_label:
            train_untarget_idx.append(i)

    train_untarget_graphs = [copy.deepcopy(graph) for graph in trainset if graph[1].item() != args.target_label]
    tmp_graphs = []
    tmp_idx = []
    num_trigger_nodes = int(avg_nodes * args.frac_of_avg)
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
    trigger_list = []
    for data in train_trigger_graphs:
        trigger_num = random.sample(data[0].nodes().tolist(), num_trigger_nodes)
        trigger_list.append(trigger_num)

    for  i, data in enumerate(train_trigger_graphs):
        for j in range(len(trigger_list[i])-1):
            for k in range(j+1, len(trigger_list[i])):
                if (data[0].has_edges_between(trigger_list[i][j], trigger_list[i][k]) or data[0].has_edges_between(trigger_list[i][k], trigger_list[i][j])) \
                    and G_trigger.has_edge(j, k) is False:
                    ids = data[0].edge_ids(torch.tensor([trigger_list[i][j], trigger_list[i][k]]), torch.tensor([trigger_list[i][k], trigger_list[i][j]]))
                    data[0].remove_edges(ids)
                elif (data[0].has_edges_between(trigger_list[i][j], trigger_list[i][k]) or data[0].has_edges_between(trigger_list[i][k], trigger_list[i][j])) is False \
                    and G_trigger.has_edge(j, k):
                    data[0].add_edges(torch.tensor([trigger_list[i][j], trigger_list[i][k]]), torch.tensor([trigger_list[i][k], trigger_list[i][j]]))
    ## rebuild data with target label
    graphs = [data[0] for data in train_trigger_graphs]
    labels = [torch.tensor([args.target_label]) for i in range(len(train_trigger_graphs))]
    train_trigger_graphs = DGLFormDataset(graphs, labels)

    test_clean_graphs = [copy.deepcopy(graph) for graph in testset]
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
    for graph in test_changed_graphs:
        trigger_idx = random.sample(graph[0].nodes().tolist(), num_trigger_nodes)
        for i in range(len(trigger_idx)-1):
            for j in range(i+1, len(trigger_idx)):
                if (graph[0].has_edges_between(trigger_idx[i], trigger_idx[j]) or graph[0].has_edges_between(trigger_idx[j], trigger_idx[i])) \
                    and G_trigger.has_edge(i, j) is False:
                    ids = graph[0].edge_ids(torch.tensor([trigger_idx[i], trigger_idx[j]]), torch.tensor([trigger_idx[j], trigger_idx[i]]))
                    graph[0].remove_edges(ids)
                elif (graph[0].has_edges_between(trigger_idx[i], trigger_idx[j]) or graph[0].has_edges_between(trigger_idx[j], trigger_idx[i])) is False \
                    and G_trigger.has_edge(i, j):
                    graph[0].add_edges(torch.tensor([trigger_idx[i], trigger_idx[j]]), torch.tensor([trigger_idx[j], trigger_idx[i]]))
    graphs = [data[0] for data in test_changed_graphs]
    labels = [torch.tensor([args.target_label]) for i in range(len(test_changed_graphs))]
    test_trigger_graphs = DGLFormDataset(graphs, labels)

    return train_trigger_graphs, test_trigger_graphs, final_idx

def inject_global_trigger_test(testset, avg_nodes, args, triggers):
    test_changed_graphs = [copy.deepcopy(graph) for graph in testset if graph[1].item() != args.target_label]
    
    num_mali = len(triggers)
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
        trigger_idx = random.sample(graph[0].nodes().tolist(), num_trigger_nodes)
        for idx, trigger in enumerate(triggers):
            start = each_trigger_nodes * idx
            for i in range(start, start+each_trigger_nodes-1):
                for j in range(i+1, start+each_trigger_nodes):
                    if (graph[0].has_edges_between(trigger_idx[i], trigger_idx[j]) or graph[0].has_edges_between(trigger_idx[j], trigger_idx[i])) \
                        and trigger.has_edge(i, j) is False:
                        ids = graph[0].edge_ids(torch.tensor([trigger_idx[i], trigger_idx[j]]), torch.tensor([trigger_idx[j], trigger_idx[i]]))
                        graph[0].remove_edges(ids)
                    elif (graph[0].has_edges_between(trigger_idx[i], trigger_idx[j]) or graph[0].has_edges_between(trigger_idx[j], trigger_idx[i])) is False \
                        and trigger.has_edge(i, j):
                        graph[0].add_edges(torch.tensor([trigger_idx[i], trigger_idx[j]]), torch.tensor([trigger_idx[j], trigger_idx[i]]))
    graphs = [data[0] for data in test_changed_graphs]
    labels = [torch.tensor([args.target_label]) for i in range(len(test_changed_graphs))]
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
        trigger_idx = random.sample(graph[0].nodes().tolist(), num_trigger_nodes)
        for idx, trigger in enumerate(triggers):
            start = each_trigger_nodes * idx
            for i in range(start, start+each_trigger_nodes-1):
                for j in range(i+1, start+each_trigger_nodes):
                    if (graph[0].has_edges_between(trigger_idx[i], trigger_idx[j]) or graph[0].has_edges_between(trigger_idx[j], trigger_idx[i])) \
                        and trigger.has_edge(i, j) is False:
                        ids = graph[0].edge_ids(torch.tensor([trigger_idx[i], trigger_idx[j]]), torch.tensor([trigger_idx[j], trigger_idx[i]]))
                        graph[0].remove_edges(ids)
                    elif (graph[0].has_edges_between(trigger_idx[i], trigger_idx[j]) or graph[0].has_edges_between(trigger_idx[j], trigger_idx[i])) is False \
                        and trigger.has_edge(i, j):
                        graph[0].add_edges(torch.tensor([trigger_idx[i], trigger_idx[j]]), torch.tensor([trigger_idx[j], trigger_idx[i]]))
    graphs = [data[0] for data in train_trigger_graphs]
    labels = [torch.tensor([args.target_label]) for i in range(len(train_trigger_graphs))]
    train_trigger_graphs = DGLFormDataset(graphs, labels)
    return train_trigger_graphs, final_idx


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

def check_graph_type(dataset):
    graph = dataset.train[0][0]
    edges = graph.edges()
    edges_0 = edges[0].tolist()
    edges_1 = edges[1].tolist()
    count = 0
    for i in range(len(edges_0)):
        for j in range(i, len(edges_0)):
            if edges_0[j] == edges_1[i] and edges_1[j] == edges_0[i]:
                count += 2
    if count == len(edges_0):
        flag = True
    else:
        flag = False
    return flag

def non_iid_split(trainset, testset, args, num_classes):
    #sort trainset
    sorted_trainset = []
    for i in range(num_classes):
        indices = [idx for idx in range(len(trainset)) if trainset[idx][1] == i]
        tmp = [trainset[j] for j in indices]
        sorted_trainset.append(tmp)
    #split data for every class
    if num_classes == 2:
        p = 0.7
    else:
        p = 0.5
    length_list = []
    for i in range(num_classes):
        n = len(sorted_trainset[i])
                                                                                                                                                                                                                                                                    
        p_list = [((1-p)*num_classes)/((num_classes-1)*args.num_workers)] * args.num_workers
        if i*args.num_workers % num_classes != 0:
            start_idx = int(i*args.num_workers/num_classes) +1
            p_list[start_idx-1] = ((1-p)*num_classes)/((num_classes-1)*args.num_workers)*(i*args.num_workers/num_classes-start_idx+1) + \
                p*num_classes/args.num_workers * (start_idx - i*args.num_workers/num_classes)
        else:
            start_idx = int(i*args.num_workers/num_classes)

        if (i+1)*args.num_workers % num_classes != 0:
            end_idx = int((i+1)*args.num_workers/num_classes)
            p_list[end_idx] = p*num_classes/args.num_workers * ((i+1)*args.num_workers/num_classes-end_idx) + \
                ((1-p)*num_classes)/((num_classes-1)*args.num_workers)*(1 - (i+1)*args.num_workers/num_classes + end_idx)
        else:
            end_idx = int(start_idx + args.num_workers/num_classes)
        
        for k in range(start_idx, end_idx):
            p_list[k] = p*num_classes/args.num_workers
        
        length = [pro * n for pro in p_list]
        length = [int(e) for e in length]
        length[-1] = n-sum(length[:-1])
        length_list.append(length)
    partition = []
    for i in range(args.num_workers):
        dataset = []
        
        for j in range(num_classes):
            start_idx = sum(length_list[j][:i])
            end_idx = start_idx + length_list[j][i]
            dataset += [sorted_trainset[j][k] for k in range(start_idx, end_idx)]
            
        partition.append(dataset)
    partition.append(testset)
    return partition


def split_dataset(args, dataset):
    num_classes = torch.max(dataset.all.graph_labels).item() + 1
    dataset_all = dataset.train[0] + dataset.val[0] + dataset.test[0]

    graph_sizes = []
    for data in dataset_all:
        graph_sizes.append(data[0].num_nodes())
    graph_sizes.sort()
    n = int(0.3*len(graph_sizes))
    graph_size_normal = graph_sizes[n:len(graph_sizes)-n]
    count = 0
    for size in graph_size_normal:
        count += size
    avg_nodes = count / len(graph_size_normal)
    avg_nodes = round(avg_nodes)

    total_size = len(dataset_all)
    test_size = int(total_size/(4*args.num_workers+1))
    train_size = total_size - test_size
    client_num = int(train_size/args.num_workers)
    length = [client_num]*(args.num_workers-1)
    length.append(train_size-(args.num_workers-1)*client_num)
    length.append(test_size)
    partition_data = random_split(dataset_all, length) # split training data and test data

    # non-iid split
    #length = [train_size, test_size]
    #trainset, testset = random_split(dataset_all, length)
    #partition_data = non_iid_split(trainset, testset, args, num_classes)
    return partition_data, avg_nodes

