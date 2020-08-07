# Execute Federated Collaborative Filtering
from FedCF import *
from torchvision import datasets, transforms
import torch
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_data():
    # data sets
    data = []
    for i in range(1, 7):
        #d = np.load("/home/jyfan/data/bank/non-iid/3clients/bank" + str(i) + ".npy")
        d = np.load("/home/jyfan/data/bank/non-iid/bank" + str(i) + ".npy")
        data.append((d[:, :16], d[:, 16:].flatten()))
    return data


def load_mnist():
    data_train = datasets.MNIST(root="~/data/", train=True, transform=transforms.ToTensor())
    data_test = datasets.MNIST(root="~/data/", train=False, transform=transforms.ToTensor())
    # split MNIST (training set) into non-iid data sets
    non_iid = []
    for i in range(0, 10):
        idx = np.where(data_train.targets == i)
        d = data_train.data[idx].flatten(1).float() / 255.0
        targets = data_train.targets[idx].float()
        non_iid.append((d, targets))
    non_iid.append((data_test.data.flatten(1).float() / 255.0, data_test.targets.float()))
    return non_iid

def mnist_svd(l):
    d = load_mnist()
    par = {
        'client_num': 100,
        'data': d,
        'device': device,
        'lr': 1,
        'server_lr': 1,
        'latent': 28,
        'lambda': l
    }
    FedCF = FederatedCF(par)
    for i in range(30):
        FedCF.global_update()
        rmse = FedCF.global_rmse()
        print("global epochs = {:d}, rmse = {:.4f}".format(i+1, rmse))
    FedCF.save_user_item_factor()


def svd_test_set(Q, epoch, lr, _lambda):
    """ Matrix Factorization for test set  """
    test_set = datasets.MNIST(root="~/data/", train=False, transform=transforms.ToTensor())
    data = test_set.data.flatten(1).float() / 255.0
    CF = MatrixFactorization(data, Q, epoch, lr, _lambda, device).to(device)
    CF.update()
    test_label = np.array(test_set.targets.float())
    np.save("/home/jyfan/data/MNIST/non-iid-p/" + str(Q.shape[0]) + "/test_label.npy", test_label)
    CF.save_user_item()


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    l = 1e-7
    #mnist_svd(l)
    Q = np.load("/home/jyfan/data/MNIST/Q_10.npy")
    svd_test_set(torch.tensor(Q), epoch=30, lr=1, _lambda=l)




