import logging
import torch
import time
from abc import ABCMeta, abstractmethod
from Common.Utils.options import args_parser
from Common.Utils.evaluate import evaluate_accuracy
#uploading gradients
logger = logging.getLogger('client.workerbase')

'''
This is the worker for sharing the local gradients.
'''
class WorkerBase(metaclass=ABCMeta):
    def __init__(self, model, loss_func, train_iter, test_iter, config, device, optimizer):
        self.model = model
        self.loss_func = loss_func

        self.train_iter = train_iter
        self.test_iter = test_iter

        self.config = config
        self.optimizer = optimizer

        # Accuracy record
        self.acc_record = [0]

        self.device = device
        self._level_length = None
        self._grad_len = 0
        self._gradients = None

    def get_gradients(self):
        """ getting gradients """
        return self._gradients

    def set_gradients(self, gradients):
        """ setting gradients """
        # try:
        #     if len(gradients) < self._grad_len:
        #         raise Exception("gradients length error!")
        # except Exception as e:
        #     logger.error(e)
        # else:
        self._gradients = gradients

    def train_step(self, x, y):
        """ Find the update gradient of each step in collaborative learning """
        x = x.to(self.device)
        y = y.to(self.device)

        y_hat = self.model(x)
        loss = self.loss_func(y_hat, y)
        self.optimizer.zero_grad()
        loss.backward()

        self._gradients = []
        self._level_length = [0]

        for param in self.model.parameters():
            self._level_length.append(param.grad.numel() + self._level_length[-1])
            self._gradients += param.grad.view(-1).cpu().numpy().tolist()

        self._grad_len = len(self._gradients)
        return loss.cpu().item(), y_hat

    def upgrade(self):
        """ Use the processed gradient to update the gradient """
        # try:
        #     if len(self._gradients) != self._grad_len:
        #         raise Exception("gradients is wrong")
        # except Exception as e:
        #     logger.error(e)

        idx = 0
        for param in self.model.parameters():
            tmp = self._gradients[self._level_length[idx]:self._level_length[idx + 1]]
            grad_re = torch.tensor(tmp, device=self.device)
            grad_re = grad_re.view(param.grad.size())

            param.grad = grad_re
            idx += 1

        self.optimizer.step()

    def train(self):
        """ General local training methods """
        self.acc_record = [0]
        for epoch in range(self.config.num_epochs):
            train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
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

            test_acc = evaluate_accuracy(self.test_iter, self.model)
            self.eva_record += [test_acc]
            # print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
            #       % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
            print(train_acc_sum / n)

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
                    print('id:',args_parser().id,'epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
                  % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
                    #print(test_acc)


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
