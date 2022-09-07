from Common.Node.workerbasev2 import WorkerBaseV2
import torch
from torch import nn
from torch import device
import json
import os
from Common.Utils.options import args_parser
from Common.Utils.gnn_util import transform_dataset, inject_global_trigger_test, save_object, split_dataset
import time
from Common.Utils.evaluate import gnn_evaluate_accuracy_v2
import numpy as np 
import torch.nn.functional as F
from GNN_common.data.TUs import TUsDataset
from GNN_common.nets.TUs_graph_classification.load_net import gnn_model  # import GNNs
from torch.utils.data import DataLoader

def server_robust_agg(args, grad): ## server aggregation
    grad_in = np.array(grad).reshape((args.num_workers, -1)).mean(axis=0)
    return grad_in.tolist()
    
class ClearDenseClient(WorkerBaseV2):
    def __init__(self, client_id, model, loss_func, train_iter, attack_iter, test_iter, config, optimizer, device, grad_stub, args):
        super(ClearDenseClient, self).__init__(model=model, loss_func=loss_func, train_iter=train_iter, attack_iter=attack_iter, test_iter=test_iter, config=config, optimizer=optimizer, device=device)
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
    time_1 = time.time()
    args = args_parser()
    with open(args.config) as f:
        config = json.load(f)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataset = TUsDataset(args.dataset)

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
    model = gnn_model(MODEL_NAME, net_params)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=args.step_size, gamma=args.gamma)
    print("Target Model:\n{}".format(model))
    client = []
    loss_func = nn.CrossEntropyLoss()
    # Load data
    partition, avg_nodes = split_dataset(args, dataset)
    drop_last = True if MODEL_NAME == 'DiffPool' else False
    triggers = []
    for i in range(args.num_workers):
        args.id = i
        train_dataset = partition[i]
        test_dataset = partition[-1]
        print("Client %d training data num: %d"%(i, len(train_dataset)))
        print("Client %d testing data num: %d"%(i, len(test_dataset)))
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                     drop_last=drop_last,
                                     collate_fn=dataset.collate)
        attack_loader = None
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,
                                     drop_last=drop_last,
                                     collate_fn=dataset.collate)
        
        client.append(ClearDenseClient(client_id=args.id, model=model, loss_func=loss_func, train_iter=train_loader, attack_iter=attack_loader, test_iter=test_loader, config=config, optimizer=optimizer, device=device, grad_stub=None, args=args))
    # prepare backdoor local backdoor dataset
    train_loader_list = []
    attack_loader_list = []
    for i in range(args.num_mali):
        train_trigger_graphs, test_trigger_graphs, G_trigger, final_idx = transform_dataset(partition[i], partition[-1], avg_nodes, args)
        triggers.append(G_trigger)
        tmp_graphs = [partition[i][idx] for idx in range(len(partition[i])) if idx not in final_idx]
        train_dataset = train_trigger_graphs + tmp_graphs

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                drop_last=drop_last,
                                collate_fn=dataset.collate)
        attack_loader = DataLoader(test_trigger_graphs, batch_size=args.batch_size, shuffle=True,
                                drop_last=drop_last,
                                collate_fn=dataset.collate)
        train_loader_list.append(train_loader)
        attack_loader_list.append(attack_loader)
    # save global trigger in order to implement centrilized backoor attack
    if args.num_mali > 0:
        filename = "./Data/global_trigger/%d/%s_%s_%d_%d_%d_%.2f_%.2f_%.2f"\
%(args.seed, MODEL_NAME, config['dataset'], args.num_workers, args.num_mali, args.epoch_backdoor, args.frac_of_avg, args.poisoning_intensity, args.density) + '.pkl'
        path = os.path.split(filename)[0]
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        save_object(triggers, filename)
        test_global_trigger = inject_global_trigger_test(partition[-1], avg_nodes, args, triggers)
        test_global_trigger_load = DataLoader(test_global_trigger, batch_size=args.batch_size, shuffle=True,
                                drop_last=drop_last,
                                collate_fn=dataset.collate)
        time_2 = time.time()
        print("saveing triggers using time: %.3f"%(time_2-time_1))
    acc_record = [0]
    counts = 0
    for epoch in range(params['epochs']):
        print('epoch:',epoch)
        if epoch >= args.epoch_backdoor:
            # malicious clients start backdoor attack
            for i in range(0, args.num_mali):
                client[i].train_iter = train_loader_list[i]
                client[i].attack_iter = attack_loader_list[i]

        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for i in range(args.num_workers):
            att_list = []
            train_loss, train_acc, test_loss, test_acc = client[i].gnn_train_v2()
            print('Client %d, loss %.4f, train acc %.3f, test loss %.4f, test acc %.3f'
                    % (i, train_loss, train_acc, test_loss, test_acc))
            if not args.filename == "":
                save_path = os.path.join(args.filename, str(args.seed), config['model'] + '_' + args.dataset + \
                    '_%d_%d_%.2f_%.2f_%.2f'%(args.num_workers, args.num_mali, args.frac_of_avg, args.poisoning_intensity, args.density) + '_%d.txt'%i)
                path = os.path.split(save_path)[0]
                isExist = os.path.exists(path)
                if not isExist:
                    os.makedirs(path)

                with open(save_path, 'a') as f:
                    f.write('%.3f %.3f %.3f %.3f'%(train_loss, train_acc, test_loss, test_acc))
                    f.write('\n')


        weights = []
        for i in range(args.num_workers):
            weights.append(client[i].get_weights())
        # Aggregation in the server to get the global model
        result = server_robust_agg(args, weights)

        for i in range(args.num_workers):
            client[i].set_weights(weights=result)
            client[i].upgrade()

        # evaluate the global model: test_acc
        test_acc = gnn_evaluate_accuracy_v2(client[0].test_iter, client[0].model)
        print('Global Test Acc: %.3f'%test_acc)
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

        # inject triggers into the testing data
        if args.num_mali > 0 and epoch >= args.epoch_backdoor:
            local_att_acc = []
            global_att_acc = gnn_evaluate_accuracy_v2(test_global_trigger_load, client[0].model)
            print('Global model with global trigger: %.3f'%global_att_acc)
            for i in range(args.num_mali):
                tmp_acc = gnn_evaluate_accuracy_v2(attack_loader_list[i], client[0].model)
                print('Global model with local trigger %d: %.3f'%(i,tmp_acc))
                local_att_acc.append(tmp_acc)

            if not args.filename == "":
                save_path = os.path.join(args.filename, str(args.seed), MODEL_NAME + '_' + args.dataset + '_%d_%d_%.2f_%.2f_%.2f'\
                           %(args.num_workers, args.num_mali, args.frac_of_avg, args.poisoning_intensity, args.density) + '_global_attack.txt')
                path = os.path.split(save_path)[0]
                isExist = os.path.exists(path)
                if not isExist:
                    os.makedirs(path)
                with open(save_path, 'a') as f:
                    f.write("%.3f" % (global_att_acc))
                    f.write(' ')
                    for i in range(args.num_mali):
                        f.write("%.3f" % (local_att_acc[i]))
                        f.write(' ')
                    f.write('\n')

