# Application of FL task
from MLModel import *
from FLModel import *
from utils import *

from torchvision import datasets, transforms
import torch
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_data():
    # data sets
    data = []
    for i in range(1, 7):
        d = np.load("/home/jyfan/data/bank/non-iid/bank" + str(i) + ".npy")
        data.append((d[:, :16], d[:, 16:].flatten()))
    return data


def load_mnist(num_users):
    # data_train = datasets.MNIST(root="~/data/", train=True, transform=transforms.ToTensor())
    data_train = datasets.MNIST(root="~/data/", train=True, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

    # data_test = datasets.MNIST(root="~/data/", train=False, transform=transforms.ToTensor())
    data_test = datasets.MNIST(root="~/data/", train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

    # split MNIST (training set) into non-iid data sets
    non_iid = []
    '''
    for i in range(0, 10):
        idx = np.where(data_train.targets == i)
        d = data_train.data[idx].flatten(1).float()
        targets = data_train.targets[idx].float()
        non_iid.append((d, targets))
    non_iid.append((data_test.data.flatten(1).float(), data_test.targets.float()))
    '''
    user_dict = mnist_noniid(data_train, num_users)
    for i in range(num_users):
        idx = user_dict[i]
        d = data_train.data[idx].flatten(1).float()
        targets = data_train.targets[idx].float()
        non_iid.append((d, targets))
    non_iid.append((data_test.data.flatten(1).float(), data_test.targets.float()))
    return non_iid


def load_p(latent):
    pth = "/home/jyfan/data/MNIST/"
    non_iid_p = []
    for i in range(10):
        d = np.load(pth + "non-iid-p/" + str(latent) + "/P_" + str(i) + ".npy")
        d = (d.T / abs(d).max(1)).T
        target = [i*1.0 for ii in range(d.shape[0])]
        non_iid_p.append((np.array(d), np.array(target)))
    test = np.load(pth + "non-iid-p/" + str(latent) + "/P10_test.npy")
    test = (test.T / abs(test).max(1)).T
    t_label = np.load(pth + "non-iid-p/" + str(latent) + "/test_label.npy")
    non_iid_p.append((test, t_label))
    return non_iid_p


if __name__ == '__main__':
    """
    1. load_data
    2. generate clients (step 3)
    3. generate aggregator
    4. training
    """
    # d = load_data()
    client_num = 10
    #d = load_mnist(client_num)
    d = load_p(latent=10)
    fl_par = {
        'output_size': 10,
        'client_num': client_num,
        'model': DeepNN,
        'data': d,
        'lr': 0.001,
        'E': 5,
        'C': 1,
        'tot_E': 100,
        'batch_size': 32,
        'device': device
    }
    import warnings
    warnings.filterwarnings("ignore")
    fl_entity = FLServer(fl_par).to(device)
    fl_entity.global_update()




