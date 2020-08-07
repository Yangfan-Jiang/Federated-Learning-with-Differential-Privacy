# Several basic machine learning models
import torch
from torch import nn

import numpy as np
import random
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LogisticRegression(nn.Module):
    """A simple implementation of Logistic regression model"""
    def __init__(self, num_feature, output_size):
        super(LogisticRegression, self).__init__()

        self.num_feature = num_feature
        self.output_size = output_size
        self.linear = nn.Linear(self.num_feature, self.output_size)
        self.sigmoid = nn.Sigmoid()
        self.model = nn.Sequential(self.linear, self.sigmoid)

    def forward(self, x):
        return self.model(x)


class DeepNN(nn.Module):
    """A simple implementation of Deep Neural Network model"""
    def __init__(self, num_feature, output_size):
        super(DeepNN, self).__init__()
        self.hidden = 200
        self.model = nn.Sequential(
            nn.Linear(num_feature, self.hidden),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(self.hidden, output_size))

    def forward(self, x):
        return self.model(x)


# test above model using ``bank" dataset
if __name__ == '__main__':
    data_pth = "/home/jyfan/data/bank/bank-full.npy"
    data = np.load(data_pth)
    tot = data.shape[0]

    x = data[:, :16]
    y = data[:, 16:]
    train_id = np.array(random.sample(range(tot), int(tot*0.8)))
    test_id = np.array(list(set(range(tot)) - set(train_id)))

    train_x = x[train_id]
    train_y = y[train_id].flatten()
    test_x = x[test_id]
    test_y = y[test_id].flatten()

    # deploy data to gpu
    train_x = torch.tensor(train_x).to(device)
    train_y = torch.tensor(train_y).to(device)
    test_x = torch.tensor(test_x).to(device)
    test_y = torch.tensor(test_y).to(device)

    # model, loss function, and optimizer
    model = LogisticRegression(16, 10).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    # training
    print("start training...")
    for epoch in range(100):
        pred_y = model(train_x)
        pred_y = pred_y.flatten()
        loss = criterion(pred_y, train_y)

        # compute acc using test set
        t_pred_y = model(test_x)
        t_pred_y = t_pred_y.flatten()
        mask = (t_pred_y > 0.5)*1.0
        acc = (mask == test_y).sum().item() / test_y.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch+1) % 1 == 0:
            print('loss = {:.4f}, acc = {:.4f}'.format(loss.data.item(), acc))
