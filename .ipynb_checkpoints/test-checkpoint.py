# Application of FL task
from MLModel import *
from FLModel import *
from utils import *

from torchvision import datasets, transforms
import torch
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_cnn_mnist(num_users):
    data_train = datasets.MNIST(root="~/data/", train=True, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

    data_test = datasets.MNIST(root="~/data/", train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

    # split MNIST (training set) into non-iid data sets
    non_iid = []
    user_dict = mnist_noniid(data_train, num_users)
    for i in range(num_users):
        idx = user_dict[i]
        d = data_train.data[idx].float().unsqueeze(1)
        targets = data_train.targets[idx].float()
        non_iid.append((d, targets))
    non_iid.append((data_test.data.float().unsqueeze(1), data_test.targets.float()))
    return non_iid

"""
1. load_data
2. generate clients (step 3)
3. generate aggregator
4. training
"""
client_num = 10
d = load_cnn_mnist(client_num)

lr = 0.1
fl_param = {
    'output_size': 10,
    'client_num': client_num,
    'model': MnistCNN,
    'data': d,
    'lr': lr,
    'E': 1,
    'C': 1,
    'eps': 4.0,
    'delta': 1e-5,
    'q': 0.03,
    'clip': 32,
    'tot_T': 150,
    'batch_size': 128,
    'device': device
}
import warnings
warnings.filterwarnings("ignore")
fl_entity = FLServer(fl_param).to(device)

print("mnist")
acc = []
for e in range(150):
    #if e+1 % 10 == 0:
    #    lr *= 0.1
    fl_entity.set_lr(lr/np.sqrt(e+1))
    acc += [fl_entity.global_update()]
    print("global epochs = {:d}, acc = {:.4f}".format(e+1, acc[-1]))