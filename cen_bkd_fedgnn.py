from Common.Node.workerbasev2 import WorkerBaseV2
import torch
from torch import nn
from torch import device
import json
import os

from Common.Utils.options import args_parser
from Common.Utils.gnn_util import inject_global_trigger_test, inject_global_trigger_train, load_pkl, split_dataset
import time
from Common.Utils.evaluate import gnn_evaluate_accuracy_v2
import numpy as np 
import torch.nn.functional as F
from GNN_common.data.TUs import TUsDataset
from GNN_common.nets.TUs_graph_classification.load_net import gnn_model  # import GNNs
from torch.utils.data import DataLoader
from defense import foolsgold, flame

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

class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


if __name__ == '__main__':
    args = args_parser()
    with open(args.config) as f:
        config = json.load(f)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataset = TUsDataset(config['dataset'])

    collate = dataset.collate
    MODEL_NAME = config['model']
    net_params = config['net_params']
    if MODEL_NAME in ['GCN', 'GAT']:
        if net_params['self_loop']:
            print("[!] Adding graph self-loops for GCN/GAT models (central node trick).")
            dataset._add_self_loops()
    params = config['params']
    net_params['in_dim'] = dataset.all.graph_lists[0].ndata['feat'][0].shape[0]
    num_classes = torch.max(dataset.all.graph_labels).item() + 1
    net_params['n_classes'] = num_classes
    net_params['dropout'] = args.dropout
    model = gnn_model(MODEL_NAME, net_params)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=args.step_size, gamma=args.gamma)

    ## set a global model
    global_model = gnn_model(MODEL_NAME, net_params)
    global_model = global_model.to(device)

    print("Target Model:\n{}".format(model))
    client = []
    loss_func = nn.CrossEntropyLoss()
    # Load data
    partition, avg_nodes = split_dataset(args, dataset)
    drop_last = True if MODEL_NAME == 'DiffPool' else False
    filename = "./Data/global_trigger/%d/%s_%s_%d_%d_%d_%.2f_%.2f_%.2f"\
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
    acc_record = [0]
    counts = 0
    weight_history = []
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
            
            if not args.filename == "":
                save_path = os.path.join(args.filename, str(args.seed), config['model'] + '_' + args.dataset + '_%d_%d_%.2f_%.2f_%.2f'\
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
        # Aggregation in the server to get the global model
        if args.defense == 'foolsgold':
            result, weight_history, alpha = foolsgold(args, weight_history, weights)
            save_path = os.path.join("./Data/alpha/%d/DBA"%(args.seed), MODEL_NAME + '_' + args.dataset + \
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
        elif args.defense == 'flame':
            result = flame(args, weights)
        else:
            result = server_robust_agg(args, weights)

        for i in range(args.num_workers):
            client[i].set_weights(weights=result)
            client[i].upgrade()
        
        # evaluate the global model: test_acc
        test_acc = gnn_evaluate_accuracy_v2(client[0].test_iter, client[0].model)
        print("Global Test acc: %.3f"%test_acc)
        if not args.filename == "":
            save_path = os.path.join(args.filename, str(args.seed), MODEL_NAME + '_' + args.dataset + '_%d_%d_%.2f_%.2f_%.2f'\
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
                save_path = os.path.join(args.filename, str(args.seed), MODEL_NAME + '_' + args.dataset + '_%d_%d_%.2f_%.2f_%.2f'%(args.num_workers, args.num_mali, args.frac_of_avg, args.poisoning_intensity, args.density) + '_global_attack.txt')
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
        
                

