import logging
import torch
import time
from abc import ABCMeta, abstractmethod
import numpy as np
from torch._C import device
from Common.Utils.options import args_parser
from Common.Utils.evaluate import evaluate_accuracy, gnn_evaluate_accuracy
import torch.nn.functional as F
from GNN_common.train.metrics import accuracy_TU as accuracy
import os
logger = logging.getLogger('client.workerbase')

'''
This is the worker for sharing the local weights.
'''
class WorkerBaseV2(metaclass=ABCMeta):
    def __init__(self, model, loss_func, train_iter, attack_iter, test_iter, config, optimizer, device):
        self.model = model
        self.loss_func = loss_func

        self.train_iter = train_iter
        self.test_iter = test_iter
        self.attack_iter = attack_iter
        self.config = config
        self.optimizer = optimizer

        # Accuracy record
        self.acc_record = [0]

        self.device = device
        self._level_length = None
        self._weights_len = 0
        self._weights = None

    def get_weights(self):
        """ getting weights """
        return self._weights

    def set_weights(self, weights):
        """ setting weights """
        # try:
        #     if len(weights) < self._weights_len:
        #         raise Exception("weights length error!")
        # except Exception as e:
        #     logger.error(e)
        # else:
        self._weights = weights

    def train_step(self, x, y):
        """ Find the update gradient of each step in collaborative learning """
        x = x.to(self.device)
        y = y.to(self.device)

        y_hat = self.model(x)
        loss = self.loss_func(y_hat, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._weights = []
        self._level_length = [0]

        for param in self.model.parameters():
            self._level_length.append(param.data.numel() + self._level_length[-1])
            self._weights += param.data.view(-1).numpy().tolist()

        self._weights_len = len(self._weights)
        return loss.cpu().item(), y_hat

    def upgrade(self):
        """ Use the processed weights to update the model """
        # try:
        #     if len(self._gradients) != self._grad_len:
        #         raise Exception("gradients is wrong")
        # except Exception as e:
        #     logger.error(e)

        idx = 0
        for param in self.model.parameters():

            tmp = self._weights[self._level_length[idx]:self._level_length[idx + 1]]
            weights_re = torch.tensor(tmp, device=self.device)
            weights_re = weights_re.view(param.data.size())

            param.data = weights_re
            idx += 1

    def train(self): # This function is for local train one epoch using local dataset on client
        """ General local training methods """
        self.acc_record = [0]
        #for epoch in range(self.config.num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        #print(self.client_id)
        for X, y in self.train_iter:
            X = X.to(self.device)
            y = y.to(self.device)
            y_hat = self.model(X)
            l = self.loss_func(y_hat, y)
            self.optimizer.zero_grad()
            l.backward()
            self.optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        
        self._weights = []
        self._level_length = [0]
        
        for param in self.model.parameters():
            self._level_length.append(param.data.numel() + self._level_length[-1])
            self._weights += param.data.view(-1).numpy().tolist()

        self._weights_len = len(self._weights)
          
        if self.client_id == 0:
            test_acc = evaluate_accuracy(self.test_iter, self.model)
            self.acc_record += [test_acc]
            print('loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
                  % (train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

    def fl_train(self, times):

        self.acc_record = [0]
        counts = 0
        for epoch in range(self.config.num_epochs):
            train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
            for X, y in self.train_iter:
                counts += 1
                if (counts % times) != 0:
                    X = X.to(self.device)
                    y = y.to(self.device)
                    y_hat = self.model(X)
                    l = self.loss_func(y_hat, y)
                    self.optimizer.zero_grad()
                    l.backward()
                    self.optimizer.step()
                    train_l_sum += l.cpu().item()
                    train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
                    n += y.shape[0]
                    batch_count += 1

                    continue

                loss, y_hat = self.train_step(X, y)

                self.update()
                self.upgrade()
                train_l_sum += loss
                train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
                n += y.shape[0]
                batch_count += 1

                if self.test_iter != None:
                    test_acc = evaluate_accuracy(self.test_iter, self.model)
                    self.acc_record += [test_acc]
                    print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
                          % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

    def write_acc_record(self, fpath, info):
        s = ""
        for i in self.acc_record:
            s += str(i) + " "
        s += '\n'
        with open(fpath, 'a+') as f:
            f.write(info + '\n')
            f.write(s)
            f.write("" * 20)

    @abstractmethod
    def update(self):
        pass

    ## GNN model training:
    def gnn_train(self): # This function is for local train one epoch using local dataset on client
        """ General local training methods """
        self.acc_record = [0]
        #for epoch in range(self.config.num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        #print(self.client_id)
        for X in self.train_iter:
            X = X.to(self.device)
            out = self.model(X)
            #l = self.loss_func(out, X.y)
            l = F.nll_loss(out, X.y)
            self.optimizer.zero_grad()
            l.backward()
            self.optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (out.argmax(dim=1)[1] == X.y).sum().cpu().item()
            n += X.num_graphs
            batch_count += 1
        #n = len(self.train_iter)
        self._weights = []
        self._level_length = [0]
        
        for param in self.model.parameters():
            self._level_length.append(param.data.numel() + self._level_length[-1])
            self._weights += param.data.view(-1).cpu().numpy().tolist()

        self._weights_len = len(self._weights)
        test_acc = gnn_evaluate_accuracy(self.test_iter, self.model) 
        if not self.args.filename == "":
            save_path = os.path.join(self.args.filename, self.args.model + '_' + self.args.dataset + '_%d.txt'%self.client_id)
            with open(save_path, 'a') as f:
                f.write("%.3f %.3f %.3f" % (train_l_sum / batch_count, train_acc_sum / n, test_acc))
                f.write("\n")
 
        print('Client %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
                  % (self.client_id, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

       
    def gnn_train_v2(self): # This function is for local train one epoch using local dataset on client
        """ General local training methods """
        self.model.train()
        self.acc_record = [0]
        #for epoch in range(self.config.num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        #print(self.client_id)
        for batch_graphs, batch_labels in self.train_iter:
            #batch_graphs = batchdata[0]
            #batch_labels = batchdata[1]
            batch_graphs = batch_graphs.to(self.device)
            batch_x = batch_graphs.ndata['feat'].to(self.device)  # num x feat
            batch_e = batch_graphs.edata['feat'].to(self.device)
            batch_labels = batch_labels.to(torch.long)
            batch_labels = batch_labels.to(self.device)
            batch_scores = self.model.forward(batch_graphs, batch_x, batch_e)
            #l = self.loss_func(out, X.y)
            l = self.model.loss(batch_scores, batch_labels)
            self.optimizer.zero_grad()
            l.backward()
            self.optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += accuracy(batch_scores, batch_labels)
            n += batch_labels.size(0)
            batch_count += 1

        self._weights = []
        self._level_length = [0]
        
        for param in self.model.parameters():
            self._level_length.append(param.data.numel() + self._level_length[-1])
            self._weights += param.data.view(-1).cpu().numpy().tolist()

        self._weights_len = len(self._weights)

        # print train acc of each client
        if self.attack_iter is not None:
            test_acc, test_l,  att_acc = self.gnn_evaluate()
            '''
            print('Client %d, loss %.4f, train acc %.3f, test loss %.4f, test acc %.3f, attack acc %.3f, time %.1f sec'
                    % (self.client_id, train_l_sum / batch_count, train_acc_sum / n, test_l, test_acc, att_acc, time.time() - start))
            if not self.args.filename == "":
                save_path = os.path.join(self.args.filename, self.config['model'] + '_' + self.args.dataset + '_%d_%d'%(self.args.num_workers, self.args.num_mali) + '_%d.txt'%self.client_id)
                with open(save_path, 'a') as f:
                    f.write("%.3f %.3f %.3f %.3f" % (train_l_sum / batch_count, train_acc_sum / n, test_acc, att_acc))
                    f.write("\n")
            '''
        else:
            test_acc, test_l = self.gnn_evaluate()
            '''
            print('Client %d, loss %.4f, train acc %.3f, test loss %.4f, test acc %.3f, time %.1f sec'
                    % (self.client_id, train_l_sum / batch_count, train_acc_sum / n, test_l, test_acc, time.time() - start))
            if not self.args.filename == "":
                save_path = os.path.join(self.args.filename, self.config['model'] + '_' + self.args.dataset + '_%d_%d'%(self.args.num_workers, self.args.num_mali) + '_%d.txt'%self.client_id)
                with open(save_path, 'a') as f:
                    f.write("%.3f %.3f %.3f" % (train_l_sum / batch_count, train_acc_sum / n, test_acc))
                    f.write("\n")
            '''
        #n = len(self.train_iter)

        '''
        if self.attack_iter is not None:
            att_acc = gnn_evaluate_accuracy_v2(self.attack_iter, self.model)
            print("attack acc: %.3f"%att_acc)
        
        if self.client_id == 0:
            test_acc = gnn_evaluate_accuracy_v2(self.test_iter, self.model)
            self.acc_record += [test_acc]
            print('loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
                  % (train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
        '''
        return train_l_sum / batch_count, train_acc_sum / n, test_l, test_acc

    def gnn_evaluate(self):
        #if device is None and isinstance(self.model, torch.nn.Module):
        #    device = list(self.model.parameters())[0].device
        acc_sum, acc_att, n, test_l_sum = 0.0, 0.0, 0, 0.0
        batch_count = 0
        for batch_graphs, batch_labels in self.test_iter:
            batch_graphs = batch_graphs.to(self.device)
            self.model.eval()
            batch_x = batch_graphs.ndata['feat'].to(self.device)
            batch_e = batch_graphs.edata['feat'].to(self.device)
            batch_labels = batch_labels.to(torch.long)
            batch_labels = batch_labels.to(self.device)
    
            batch_scores = self.model.forward(batch_graphs, batch_x, batch_e)
            l = self.loss_func(batch_scores, batch_labels)
            acc_sum += accuracy(batch_scores, batch_labels)
            test_l_sum += l.detach().item()
            n += batch_labels.size(0)
            batch_count += 1
            if self.attack_iter is not None:
                n_att = 0
                for batch_graphs, batch_labels in self.attack_iter:
                    batch_graphs = batch_graphs.to(self.device)
                    self.model.eval()
                    batch_x = batch_graphs.ndata['feat'].to(self.device)
                    batch_e = batch_graphs.edata['feat'].to(self.device)
                    batch_labels = batch_labels.to(self.device)

                    batch_scores = self.model.forward(batch_graphs, batch_x, batch_e)
                    acc_att += accuracy(batch_scores, batch_labels)
                    self.model.train()
                    n_att += batch_labels.size(0)
                return acc_sum / n, test_l_sum / batch_count, acc_att / n_att

        return acc_sum / n, test_l_sum / batch_count
 
