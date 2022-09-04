import sys
sys.path.append("/home/nfs/jxu8/federated_learning_jx/federated_learning/GNN_common/data")
from Common.Node.workerbasev2 import WorkerBaseV2
#from Common.Node.workerbase import WorkerBase
import torch
from torch import nn
from torch import device
import Common.config as config
import json
import os

#from Common.Model.LeNet import LeNet
#from Common.Model.ResNet import ResNet, BasicBlock
#from Common.Model.gnn_models import GIN
#from Common.Utils.data_loader import load_data_fmnist, load_data_cifar10, load_data_mnist, load_data_tud_v2
#from Common.Utils.set_log import setup_logging
from Common.Utils.options import args_parser
from Common.Utils.gnn_util import inject_global_trigger_test, inject_global_trigger_train, load_pkl
#import grpc
#import copy
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
from defense import foolsgold_jx

def server_robust_agg(args, grad): ## server aggregation
    grad_in = np.array(grad).reshape((args.num_workers, -1)).mean(axis=0)
    return grad_in.tolist()
    
class ClearDenseClient(WorkerBaseV2):
    def __init__(self, client_id, model, loss_func, train_iter, attack_iter, test_iter, config, optimizer, device, grad_stub, args):
        super(ClearDenseClient, self).__init__(model=model, loss_func=loss_func, train_iter=train_iter, attack_iter=attack_iter,
                                               test_iter=test_iter, config=config, optimizer=optimizer, device=device)
        self.client_id = client_id
        self.grad_stub = None
        self.args = args

    def update(self):
        pass
        #gradients = super().get_gradients()
        #return gradients
        #res_grad_upd = self.grad_stub.UpdateGrad_float.future(GradRequest_float(id=self.client_id, grad_ori=gradients))

        #super().set_gradients(gradients=res_grad_upd.result().grad_upd)

class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self

def split_dataset(args, dataset):
    split_number = random.randint(0, 0)
    #trainset, valset, testset = dataset.train[split_number], dataset.val[split_number], dataset.test[split_number]
    #print("Size of Original Trainset:{}, Valset:{}, Testset:{}".format(len(trainset), len(valset), len(testset)))
    #dataset_all = trainset + valset + testset
    if args.dataset == 'MOLT-4H':
        dataset_all = dataset.data_balance[0]
    else:
        dataset_all = dataset.train[0] + dataset.val[0] + dataset.test[0]
    #dataset_all = dataset.data_balance[0]
    # compute average nodes
    num_nodes = 0
    for data in dataset_all:
        num_nodes += data[0].num_nodes()
    avg_nodes = round(num_nodes / len(dataset_all))

    total_size = len(dataset_all)
    # resize the dataset into defined trainset and testset
    test_size = int(total_size / (1+4*args.num_workers))
    train_size = total_size - test_size
    #partition_data = [None]*args.num_workers
    client_num = int(train_size/args.num_workers)
    length = [client_num]*(args.num_workers-1)
    length.append(train_size-(args.num_workers-1)*client_num)
    length.append(test_size)

    partition_data = random_split(dataset_all, length) # split training data and test data
    #tmp = []
    #for i in range(len(partition_data)):
    #    train_dataset, test_dataset = split_train_test_dataset(partition_data[i])
    #    tmp.append(TrianTest(train_dataset, test_dataset))
    #partition_data = random_split(trainset, length)
    
    return partition_data, avg_nodes



if __name__ == '__main__':
    args = args_parser()
    with open(args.config) as f:
        config = json.load(f)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataset = TUsDataset(config['dataset'])
    #args.filename = '/home/nfs/jxu8/federated_learning_jx/federated_learning/results/CBA'
    '''
    # verify whether graph in dataset is undirected or not
    flag = 0
    for data in dataset.all:
        count = 0
        edges = data[0].edges()
        for i in range(len(edges[0])):
            if edges[0][i] < edges[1][i]:
                for j in range(i, len(edges[0])):
                    if edges[0][j] == edges[1][i] and edges[1][j] == edges[0][i]:
                        count = count + 2
        if count == len(edges[0]):
            flag += 1
    print(len(dataset.all), flag)
    '''

    collate = dataset.collate
    MODEL_NAME = config['model']
    net_params = config['net_params']
    if MODEL_NAME in ['GCN', 'GAT']:
        if net_params['self_loop']:
            print("[!] Adding graph self-loops for GCN/GAT models (central node trick).")
            dataset._add_self_loops()
    params = config['params']
    net_params['in_dim'] = dataset.all.graph_lists[0].ndata['feat'][0].shape[0]
    #num_classes = len(np.unique(dataset.all.graph_labels))
    num_classes = torch.max(dataset.all.graph_labels).item() + 1
    net_params['n_classes'] = num_classes
    net_params['dropout'] = args.dropout
    model = gnn_model(MODEL_NAME, net_params)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=args.step_size, gamma=args.gamma)

    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
    #                                                   factor=params['lr_reduce_factor'],
    #                                                   patience=params['lr_schedule_patience'],
    #                                                   verbose=True)
    ## set a global model
    global_model = gnn_model(MODEL_NAME, net_params)
    global_model = global_model.to(device)

    print("Target Model:\n{}".format(model))
    client = []
    loss_func = nn.CrossEntropyLoss()
    # Load data
    partition, avg_nodes = split_dataset(args, dataset)
    drop_last = True if MODEL_NAME == 'DiffPool' else False
    #filename = "/home/jxu8/Code/federated_learning_jx/federated_learning/Data/global_trigger/%s_%s"%(MODEL_NAME, config['dataset']) + '.pkl'
    filename = "/tudelft.net/staff-umbrella/GS/Graph_Neural_Networks/federated_learning_jx/data/global_trigger/foolsgold/%d/%s_%s_%d_%d_%d_%.2f_%.2f_%.2f"\
              %(args.seed, MODEL_NAME, config['dataset'], args.num_workers, args.num_mali, args.epoch_backdoor, args.frac_of_avg, args.poisoning_intensity, args.density) + '.pkl'
    global_trigger = load_pkl(filename)
    print("Triggers loaded!")
    args.num_mali = len(global_trigger)
    for i in range(args.num_workers):
        args.id = i
        print("Client %d training data num: %d"%(i, len(partition[i])))
        print("Client %d testing data num: %d"%(i, len(partition[-1])))
        train_loader = DataLoader(partition[i], batch_size=args.batch_size, shuffle=True,
                                    drop_last=drop_last,
                                    collate_fn=dataset.collate)
        attack_loader = None
        test_loader = DataLoader(partition[-1], batch_size=args.batch_size, shuffle=True,
                                    drop_last=drop_last,
                                    collate_fn=dataset.collate)
        
        client.append(ClearDenseClient(client_id=args.id, model=model, loss_func=loss_func, train_iter=train_loader, attack_iter=attack_loader, test_iter=test_loader, config=config, optimizer=optimizer, device=device, grad_stub=None, args=args))
    # prepare backdoor training dataset and testing dataset
    train_trigger_graphs, final_idx = inject_global_trigger_train(partition[0], avg_nodes, args, global_trigger)
    test_trigger_graphs = inject_global_trigger_test(partition[-1], avg_nodes, args, global_trigger)
    tmp_graphs = [partition[0][idx] for idx in range(len(partition[0])) if idx not in final_idx]

    train_dataset = train_trigger_graphs + tmp_graphs
    backdoor_train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                            drop_last=drop_last,
                            collate_fn=dataset.collate)
    backdoor_attack_loader = DataLoader(test_trigger_graphs, batch_size=args.batch_size, shuffle=True,
                            drop_last=drop_last,
                            collate_fn=dataset.collate)
    test_local_trigger_load = []
    for i in range(len(global_trigger)):
        test_local_trigger = inject_global_trigger_test(partition[-1], avg_nodes, args, [global_trigger[i]])
        tmp_load = DataLoader(test_local_trigger, batch_size=args.batch_size, shuffle=True,
                            drop_last=drop_last,
                            collate_fn=dataset.collate)
        test_local_trigger_load.append(tmp_load)
    #import pdb
    #pdb.set_trace()
    acc_record = [0]
    counts = 0
    weight_history = []
    #for epoch in range(config.num_epochs):
    for epoch in range(params['epochs']):
        print('epoch:',epoch)
        if epoch >= args.epoch_backdoor:
            # inject global trigger into the centrilized attacker - client[0]
            client[0].train_iter = backdoor_train_loader
            client[0].attack_iter = backdoor_attack_loader
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        weights = []
        for i in range(args.num_workers):
            att_list = []
            train_loss, train_acc, test_loss, test_acc = client[i].gnn_train_v2()
            global_att = gnn_evaluate_accuracy_v2(backdoor_attack_loader, client[i].model)
            print('Client %d, loss %.4f, train acc %.3f, test loss %.4f, test acc %.3f'
                    % (i, train_loss, train_acc, test_loss, test_acc))
            print('Client %d with global trigger: %.3f'%(i, global_att))
            for j in range(len(global_trigger)):
                tmp_acc = gnn_evaluate_accuracy_v2(test_local_trigger_load[j], client[i].model)
                print('Client %d with local trigger %d: %.3f'%(i, j, tmp_acc))
                att_list.append(tmp_acc)
            weights.append(client[i].get_weights())
            weight_history.append(client[i].get_weights())
            
            if not args.filename == "":
                save_path = os.path.join(args.filename, config['model'] + '_' + args.dataset + '_%d_%d_%.2f_%.2f_%.2f'\
                          %(args.num_workers, args.num_mali, args.frac_of_avg, args.poisoning_intensity, args.density) + '_%d.txt'%i)
                path = os.path.split(save_path)[0]
                isExist = os.path.exists(path)
                if not isExist:
                    os.makedirs(path)
                with open(save_path, 'a') as f:
                    f.write('%.3f %.3f %.3f %.3f %.3f '%(train_loss, train_acc, test_loss, test_acc, global_att))
                    for i in range(len(global_trigger)):
                        f.write('%.3f'%att_list[i])
                        f.write(' ')
                    f.write('\n')
        '''
        # Attack rate in each local model
        for i in range(args.num_workers):
            test_acc, att_acc = client[i].gnn_evaluate()
            print("Client: %d, Test acc: %.3f, Attack acc: %.3f"%(i, test_acc, att_acc))
        '''
        # Aggregation in the server to get the global model
        #result = server_robust_agg(args, weights)
        result, weight_history, alpha = foolsgold_jx(args, weight_history, weights)
        save_path = os.path.join("/tudelft.net/staff-umbrella/GS/Graph_Neural_Networks/federated_learning_jx/data/alpha/%d/CBA" \
                    %(args.seed), MODEL_NAME + '_' + args.dataset + \
                    '_%d_%d_%.2f_%.2f_%.2f'%(args.num_workers, args.num_mali, args.frac_of_avg, args.poisoning_intensity, args.density) + '_alpha.txt')
        path = os.path.split(save_path)[0]
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        with open(save_path, 'a') as f:
            for i in range(args.num_workers):
                f.write("%.3f" % (alpha[i]))
                f.write(' ')
            f.write("\n")

        for i in range(args.num_workers):
            client[i].set_weights(weights=result)
            client[i].upgrade()
        
        # evaluate the global model: test_acc
        test_acc = gnn_evaluate_accuracy_v2(client[0].test_iter, client[0].model)
        print("Global Test acc: %.3f"%test_acc)
        if not args.filename == "":
            save_path = os.path.join(args.filename, MODEL_NAME + '_' + args.dataset + '_%d_%d_%.2f_%.2f_%.2f'\
                       %(args.num_workers, args.num_mali, args.frac_of_avg, args.poisoning_intensity, args.density) + '_global_test.txt')
            path = os.path.split(save_path)[0]
            isExist = os.path.exists(path)
            if not isExist:
                os.makedirs(path)

            with open(save_path, 'a') as f:
                f.write("%.3f" % (test_acc))
                f.write("\n")
        if epoch >= args.epoch_backdoor:
            local_att_acc = []
            global_att_acc = gnn_evaluate_accuracy_v2(backdoor_attack_loader, client[0].model)
            print('Global model with global trigger: %.3f'%global_att_acc)
            for i in range(len(global_trigger)):
                tmp_acc = gnn_evaluate_accuracy_v2(test_local_trigger_load[i], client[0].model)
                print('Global model with local trigger %d: %.3f'%(i, tmp_acc))
                local_att_acc.append(tmp_acc)
            if not args.filename == "":
                save_path = os.path.join(args.filename, MODEL_NAME + '_' + args.dataset + '_%d_%d_%.2f_%.2f_%.2f'%(args.num_workers, args.num_mali, args.frac_of_avg, args.poisoning_intensity, args.density) + '_global_attack.txt')
                path = os.path.split(save_path)[0]
                isExist = os.path.exists(path)
                if not isExist:
                    os.makedirs(path)
                with open(save_path, 'a') as f:
                    f.write("%.3f" % (global_att_acc))
                    f.write(' ')
                    for i in range(len(global_trigger)):
                        f.write("%.3f" % (local_att_acc[i]))
                        f.write(' ')
                    f.write('\n')
        
                

