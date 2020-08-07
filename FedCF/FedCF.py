# Federated Collaborative Filtering (PyTorch version)
import torch
from torch import nn

import numpy as np


class LocalFCFModel(nn.Module):
    """
    Local Model of Federated Funk SVD, all private rating record and parameters are stored at client device.
    Each client take their own rating record and user factor vector.
    Item factor vector is shared by all clients, and should NOT be update at local device (client)
    """
    def __init__(self, loc_lr, features, loc_rating, _lambda, device=None):
        super(LocalFCFModel, self).__init__()
        self.device = device
        self.lr = loc_lr
        self.Lambda = _lambda
        self.item_factor = None
        self.rating = torch.tensor(loc_rating).to(self.device)
        self.size = self.rating.shape[0]    # number of data records
        self.user_factor = nn.Parameter(nn.init.normal_(torch.empty(self.size, features), std=0.35),
                                        requires_grad=True)
        self.H = []  # history of gradients (y')

    def forward(self, item_factor):
        """
        Input the server model (item factor vector)
        Return the prediction of ratings with shape [1, # movies]
        """
        return self.user_factor.mm(item_factor)

    def loss_obj(self):
        """Calculate loss of local model"""
        loss = ((self.forward(self.item_factor) - self.rating)**2).sum()
        regularization = self.Lambda * (torch.norm(self.user_factor)**2 +
                                        torch.norm(self.item_factor, dim=0)**2)
        return (loss + regularization.sum()) / self.size

    def recv_item_factor(self, item_factor):
        """Get the shared item factor"""
        self.item_factor = item_factor.clone().detach().to(self.device)
        self.item_factor.requires_grad_(True)

    def loc_rmse(self):
        loss = ((self.forward(self.item_factor) - self.rating)**2).sum() / self.size
        return np.sqrt(loss.detach().cpu())

    def get_user_factor(self):
        return self.user_factor.detach().cpu()

    def add_his_grad(self, grad):
        """Should be invoke after finishing local iterations"""
        self.H.append(grad.detach().cpu().numpy())


class ServerFCFModel(nn.Module):
    """
    Server Model of Federated Funk SVD
    The item factor vector is stored at server and send to all clients.
    The gradients are aggregated after all clients completing local update.
    Item factor vector is updated via this gradient
    """
    def __init__(self, srv_lr, item_num, features, device=None):
        super(ServerFCFModel, self).__init__()
        self.lr = srv_lr
        self.device = device
        self.features = features
        self.item_factor = nn.init.normal_(torch.empty(self.features, item_num), std=0.35).to(self.device)

    def update(self, grad):
        """
        Update the shard item factor vector
        """
        grad = grad.to(self.device)
        self.item_factor = self.item_factor - self.lr * grad

    def get_item_factor(self):
        return self.item_factor.detach()

    def forward(self):
        pass


class FederatedCF(object):
    def __init__(self, par):
        self.client_num = par['client_num']
        self.data = par['data']          # data set, 4D : dim=(# clients, train/test, # record, # feature)
        self.E = 10                      # each client perform E-round iteration between each communication round
        self.feature_num = self.data[0][0].shape[1]

        self.clients = None              # a list of client models
        self.server_model = None         # the server model
        self.weight = None               # size of data of each client
        self.Lambda = par['lambda']      # penalty factor
        self.device = par['device']
        self.latent = par['latent']      # dimension of latent factor is 5 by default
        self.client_lr = par['lr']
        self.server_lr = par['server_lr']
        self.init_model()
        print('feature num:', self.latent)

    def init_model(self):
        """
        Initialize model (local user factor vectors and a shared item factor vector)
        Generate a shared item factor vectors, the user factor vectors should be generated independently
        """
        self.clients = [LocalFCFModel(self.client_lr, self.latent,
                                      self.data[uid][0], self.Lambda, self.device).to(self.device)
                        for uid in range(self.client_num)]
        self.server_model = ServerFCFModel(self.server_lr, self.feature_num, self.latent, self.device).to(self.device)
        self.weight = [client.size for client in self.clients]
        self.broad_cast()

    def client_update(self, uid):
        """
        Update user factor vectors user_factor[uid] for a client
        Return gradients of item factor vector to server
        """
        self.clients[uid].train()
        optimizer = torch.optim.SGD(self.clients[uid].parameters(), lr=self.client_lr, momentum=0.9)
        # optimizer = torch.optim.Adam(self.clients[uid].parameters(), lr=self.client_lr)
        for i in range(self.E):
            optimizer.zero_grad()
            loss = self.clients[uid].loss_obj()
            loss.backward()
            optimizer.step()
        return self.clients[uid].item_factor.grad.clone().detach()

    def global_update(self):
        """Perform one round of global update.
           The gradients are aggregated via FedAvg.
        """
        grads = []
        for uid in range(self.client_num):
            w = self.weight[uid] / sum(self.weight)
            grads.append(self.client_update(uid) * w)
            torch.cuda.empty_cache()

        grads = sum(grads)
        self.server_model.update(grads)
        self.broad_cast()

    def broad_cast(self):
        """Broadcast new item factor to all clients"""
        item_factor = self.server_model.get_item_factor()
        for client in self.clients:
            client.recv_item_factor(item_factor)

    def global_rmse(self):
        """Calculate global loss"""
        loss = 0.0
        for client in self.clients:
            client.eval()
            loss += client.loc_rmse()
        return loss / self.client_num

    def save_user_item_factor(self):
        """Save P^(i) and Q"""
        p = []
        l = []
        for uid in range(self.client_num):
            client = self.clients[uid]
            p.append(client.user_factor.detach().cpu().numpy())
            np.save("/home/jyfan/data/MNIST/non-iid-p/" + str(self.latent) + "/P_" + str(uid) + ".npy",
                    client.user_factor.detach().cpu().numpy())
        for uid in range(self.client_num):
            l.append(np.array(self.data[uid][1]))
        np.save("/home/jyfan/data/MNIST/P_"+str(self.latent)+".npy", np.vstack(p))
        np.save("/home/jyfan/data/MNIST/Q_"+str(self.latent)+".npy", self.server_model.get_item_factor().cpu().numpy())
        np.save("/home/jyfan/data/MNIST/label.npy", np.hstack(l))


class MatrixFactorization(nn.Module):
    """ Compute User Latent Matrix P (Item Latent Matrix Q is fixed) """
    def __init__(self, data, Q, epoch, lr, _lambda, device):
        super(MatrixFactorization, self).__init__()
        self.lr = lr
        self.epoch = epoch
        self.data = data.to(device)
        self.Q = Q.to(device)
        self.latent = Q.shape[0]
        self.size = data.shape[0]
        self.Lambda = _lambda
        self.P = nn.Parameter(nn.init.normal_(torch.empty(self.size, self.latent), std=0.35), requires_grad=True)

    def forward(self):
        """ Compute R ~ P*Q """
        return self.P.mm(self.Q)

    def loss_obj(self):
        """ Compute loss of local model """
        loss = ((self.forward() - self.data)**2).sum()
        regularization = self.Lambda * (torch.norm(self.P)**2 +
                                        torch.norm(self.Q, dim=0)**2)
        return (loss + regularization.sum()) / self.size

    def loc_rmse(self):
        """ Compute Root Mean Square Error (RMSE) """
        loss = ((self.forward() - self.data)**2).sum() / self.size
        return np.sqrt(loss.detach().cpu())

    def update(self):
        """ Update User Latent Matrix P """
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        for i in range(self.epoch):
            optimizer.zero_grad()
            loss = self.loss_obj()
            loss.backward()
            optimizer.step()
            print("rmse = {:.4f}".format(self.loc_rmse().item()))

    def save_user_item(self):
        np.save("/home/jyfan/data/MNIST/non-iid-p/" + str(self.latent) + "/P" + str(self.latent)+"_test.npy",
                self.P.detach().cpu().numpy())


