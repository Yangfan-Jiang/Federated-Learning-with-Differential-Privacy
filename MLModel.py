# Several basic machine learning models
import torch
from torch import nn


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


class MLP(nn.Module):
    """A simple implementation of Deep Neural Network model"""
    def __init__(self, num_feature, output_size):
        super(MLP, self).__init__()
        self.hidden = 200
        self.model = nn.Sequential(
            nn.Linear(num_feature, self.hidden),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(self.hidden, output_size))

    def forward(self, x):
        return self.model(x)

