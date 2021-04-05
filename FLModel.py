# Federated Learning Model in PyTorch
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from DPMechanisms import gaussian_noise
from tensorflow_privacy.privacy.analysis.compute_noise_from_budget_lib import compute_noise

import numpy as np
import copy


class FLClient(nn.Module):
    """ Client of Federated Learning framework.
        1. Receive global model from server
        2. Perform local training (compute gradients)
        3. Return local model (gradients) to server
    """
    def __init__(self, model, output_size, data, lr, E, T, batch_size, q, clip, eps, delta, device=None):
        """
        :param model: ML model's training process should be implemented
        :param data: (tuple) dataset, all data in client side is used as training data
        :param lr: learning rate
        :param E: epoch of local update
        """
        super(FLClient, self).__init__()
        self.device = device
        self.BATCH_SIZE = batch_size
        self.torch_dataset = TensorDataset(torch.tensor(data[0]),
                                           torch.tensor(data[1]))
        self.data_size = len(self.torch_dataset)
        self.data_loader = DataLoader(
            dataset=self.torch_dataset,
            batch_size=self.BATCH_SIZE,
            shuffle=True
        )
        self.noise = None
        self.lr = lr
        self.E = E
        self.T = T
        self.clip = clip
        self.eps = eps
        self.delta = delta
        self.q = q
        self.model = model(data[0].shape[1], output_size).to(self.device)
        self.batch_model = model(data[0].shape[1], output_size).to(self.device)
        self.recv_model = model(data[0].shape[1], output_size).to(self.device)

        # compute noise using moments accountant
        self.sigma = compute_noise(1, self.q, self.eps, self.T, self.delta, 1e-5)

    def recv(self, model_param):
        """receive global model from aggregator (server)"""
        self.model.load_state_dict(copy.deepcopy(model_param))
        self.recv_model.load_state_dict(copy.deepcopy(model_param))

    def update(self):
        """local model update"""
        self.model.train()
        criterion = nn.CrossEntropyLoss(reduction='none')
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.0)

        # randomly select q fraction samples from data
        # according to the privacy analysis of moments accountant
        # training "Lots" are sampled by poisson sampling
        idx = np.where(np.random.rand(len(self.torch_dataset[:][0])) < self.q)[0]
        # sample_size = int(self.q * self.data_size)
        # idx = np.random.choice(self.data_size, sample_size, replace=False)
        sampled_dataset = TensorDataset(self.torch_dataset[idx][0], self.torch_dataset[idx][1])
        sample_data_loader = DataLoader(
            dataset=sampled_dataset,
            batch_size=self.BATCH_SIZE,
            shuffle=True
        )
        clipped_grads = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}
        for e in range(self.E):
            optimizer.zero_grad()
            for batch_x, batch_y in sample_data_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred_y = self.model(batch_x)
                loss = criterion(pred_y, batch_y.long())
                # bound l2 sensitivity (gradient clipping)
                # clip each of the gradient in the "Lot"
                for i in range(loss.size()[0]):
                    loss[i].backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip)
                    for name, param in self.model.named_parameters():
                        clipped_grads[name] += param.grad / len(idx)
                    self.model.zero_grad()
            for name, param in self.model.named_parameters():
                param.grad = clipped_grads[name]
            optimizer.step()

        # Add Gaussian noise
        # 1. compute l2-sensitivity.
        # 2. add gaussian noise
        # The sensitivity calculation formula used below is derived from an unpublished manuscript
        # Please derive and compute the l1/l2-sensitivity very carefully
        # Do not use the sensitivity calculation code below directly on any research experiments
        sensitivity = 2.0 * self.lr * self.clip / len(idx) + (self.E - 1) * 2 * self.lr * self.clip
        new_param = copy.deepcopy(self.model.state_dict())
        for name in new_param:
            new_param[name] = torch.zeros(new_param[name].shape).to(self.device)
            new_param[name] += 1.0 * self.model.state_dict()[name]
            new_param[name] += gaussian_noise(self.model.state_dict()[name].shape, sensitivity,
                                                 self.sigma, device=self.device)
        self.model.load_state_dict(copy.deepcopy(new_param))

    def update_grad(self):
        """FedSGD algorithm, return gradient"""
        self.model.train()
        criterion = nn.CrossEntropyLoss(reduction='none')
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0)

        # randomly select q fraction samples from data
        # according to the privacy analysis of moments accountant
        # training "Lots" are sampled by poisson sampling
        idx = np.where(np.random.rand(len(self.torch_dataset[:][0])) < self.q)[0]
        # sample_size = int(self.q * self.data_size)
        # idx = np.random.choice(self.data_size, sample_size, replace=False)
        sampled_dataset = TensorDataset(self.torch_dataset[idx][0], self.torch_dataset[idx][1])
        sample_data_loader = DataLoader(
            dataset=sampled_dataset,
            batch_size=self.BATCH_SIZE,
            shuffle=True
        )
        clipped_grads = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}
        optimizer.zero_grad()
        for batch_x, batch_y in sample_data_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            pred_y = self.model(batch_x)
            loss = criterion(pred_y, batch_y.long())
            # bound l2 sensitivity (gradient clipping)
            # clip each of the gradient in the "Lot"
            for i in range(loss.size()[0]):
                loss[i].backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip)
                for name, param in self.model.named_parameters():
                    clipped_grads[name] += param.grad / len(idx)
                self.model.zero_grad()
        # Add Gaussian noise
        sensitivity = 2 * self.clip / len(idx)
        for name, param in self.model.named_parameters():
            clipped_grads[name] += gaussian_noise(clipped_grads[name].shape, sensitivity, self.sigma, device=self.device)
        return clipped_grads.copy()


class FLServer(nn.Module):
    """ Server of Federated Learning
        1. Receive model (or gradients) from clients
        2. Aggregate local models (or gradients)
        3. Compute global model, broadcast global model to clients
    """
    def __init__(self, fl_param):
        super(FLServer, self).__init__()
        self.device = fl_param['device']
        self.client_num = fl_param['client_num']
        self.C = fl_param['C']  # (float) C in [0, 1]
        self.epsilon = fl_param['eps']  # (ε, δ)-DP
        self.delta = fl_param['delta']
        self.clip = fl_param['clip']
        self.T = fl_param['tot_T']  # total number of global iterations

        # For FEMnist dataset
        self.data = []
        self.target = []
        for sample in fl_param['data'][self.client_num:]:
            self.data += [torch.tensor(sample[0]).to(self.device)]  # test set
            self.target += [torch.tensor(sample[1]).to(self.device)]  # target label

        self.input_size = int(self.data[0].shape[1])
        self.lr = fl_param['lr']

        self.clients = [FLClient(fl_param['model'],
                                 fl_param['output_size'],
                                 fl_param['data'][i],
                                 fl_param['lr'],
                                 fl_param['E'],
                                 fl_param['tot_T'],
                                 fl_param['batch_size'],
                                 fl_param['q'],
                                 fl_param['clip'],
                                 fl_param['eps'],
                                 fl_param['delta'],
                                 self.device)
                        for i in range(self.client_num)]
        self.global_model = fl_param['model'](self.input_size, fl_param['output_size']).to(self.device)
        self.weight = np.array([client.data_size * 1.0 for client in self.clients])
        self.broadcast(self.global_model.state_dict())

    def aggregated(self, idxs_users):
        """FedAvg"""
        model_par = [self.clients[idx].model.state_dict() for idx in idxs_users]
        new_par = copy.deepcopy(model_par[0])
        for name in new_par:
            new_par[name] = torch.zeros(new_par[name].shape).to(self.device)
        for idx, par in enumerate(model_par):
            w = self.weight[idxs_users[idx]] / np.sum(self.weight[:])
            for name in new_par:
                # new_par[name] += par[name] * (self.weight[idxs_users[idx]] / np.sum(self.weight[idxs_users]))
                new_par[name] += par[name] * (w / self.C)
        self.global_model.load_state_dict(copy.deepcopy(new_par))
        return self.global_model.state_dict().copy()

    def broadcast(self, new_par):
        """Send aggregated model to all clients"""
        for client in self.clients:
            client.recv(new_par.copy())

    def test_acc(self):
        # compute accuracy using test set
        self.global_model.eval()
        t_pred_y = self.global_model(self.data)
        _, predicted = torch.max(t_pred_y, 1)

        acc = (predicted == self.target).sum().item() / self.target.size(0)
        return acc

    def test_acc_femnist(self):
        self.global_model.eval()
        correct = 0
        tot_sample = 0
        for i in range(len(self.data)):
            t_pred_y = self.global_model(self.data[i])
            _, predicted = torch.max(t_pred_y, 1)
            correct += (predicted == self.target[i]).sum().item()
            tot_sample += self.target[i].size(0)
        acc = correct / tot_sample
        return acc

    def global_update(self):
        idxs_users = np.random.choice(range(len(self.clients)), int(self.C * len(self.clients)), replace=False)
        for idx in idxs_users:
            self.clients[idx].update()
        self.broadcast(self.aggregated(idxs_users))
        # acc = self.test_acc()
        acc = self.test_acc_femnist()
        return acc

    def aggregated_grad(self, idxs_users, grads):
        """FedAvg - Update model using gradients"""
        agg_grad = copy.deepcopy(grads[0])
        for name in agg_grad:
            agg_grad[name] = torch.zeros(agg_grad[name].shape).to(self.device)

        for idx, grad in enumerate(grads):
            w = self.weight[idxs_users[idx]] / np.sum(self.weight[idxs_users])
            for name in grad:
                g = grad[name] * self.lr
                agg_grad[name] += g * (w / self.C)

        for name in agg_grad:
            self.global_model.state_dict()[name] -= agg_grad[name]
        return self.global_model.state_dict().copy()

    def global_update_grad(self):
        idxs_users = np.random.choice(range(len(self.clients)), int(self.C * len(self.clients)), replace=False)
        grads = []
        for idx in idxs_users:
            grads.append(copy.deepcopy(self.clients[idx].update_grad()))
        self.broadcast(self.aggregated_grad(idxs_users, grads))
        acc = self.test_acc_femnist()
        return acc

    def set_lr(self, lr):
        for c in self.clients:
            c.lr = lr
