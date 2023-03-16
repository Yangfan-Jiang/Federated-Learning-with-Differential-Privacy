# Machine learning models
import torch
from torch import nn
from kymatio.torch import Scattering2D

class MNIST_CNN(nn.Module):
    """
    End-to-end CNN model for MNIST and Fashion-MNIST, with Tanh activations. 
    References:
    - Papernot, Nicolas, et al. Tempered Sigmoid Activations for Deep Learning with Differential Privacy. In AAAI 2021.
    - Tramer, Florian, and Dan Boneh. Differentially Private Learning Needs Better Features (or Much More Data). In ICLR 2021. 
    """
    def __init__(self, input_dim, output_dim):
        super(MNIST_CNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=2, padding=2),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=1))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=1))
        
        self.fc = nn.Sequential(nn.Linear(4 * 4 * 32, 32),
                                        nn.Tanh(),
                                        nn.Linear(32, 10))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    
def get_scatter_transform():
    shape = (28, 28, 1)
    scattering = Scattering2D(J=2, shape=shape[:2])
    K = 81 * shape[2]
    (h, w) = shape[:2]
    return scattering, K, (h//4, w//4)


class ScatterLinear(nn.Module):
    """
    ScatterNet model used in the following paper
    - Tramer, Florian, and Dan Boneh. Differentially Private Learning Needs Better Features (or Much More Data). In ICLR 2021. 
    See https://github.com/ftramer/Handcrafted-DP/blob/main/models.py
    """
    def __init__(self, in_channels, hw_dims, input_norm=None, classes=10, clip_norm=None, **kwargs):
        super(ScatterLinear, self).__init__()
        self.K = in_channels
        self.h = hw_dims[0]
        self.w = hw_dims[1]
        self.fc = None
        self.norm = None
        self.clip = None
        self.build(input_norm, classes=classes, clip_norm=clip_norm, **kwargs)

    def build(self, input_norm=None, num_groups=None, bn_stats=None, clip_norm=None, classes=10):
        self.fc = nn.Linear(self.K * self.h * self.w, classes)

        if input_norm is None:
            self.norm = nn.Identity()
        elif input_norm == "GroupNorm":
            self.norm = nn.GroupNorm(num_groups, self.K, affine=False)
        else:
            self.norm = lambda x: standardize(x, bn_stats)

        if clip_norm is None:
            self.clip = nn.Identity()
        else:
            self.clip = ClipLayer(clip_norm)

    def forward(self, x):
        x = self.norm(x.view(-1, self.K, self.h, self.w))
        x = self.clip(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

    
class LogisticRegression(nn.Module):
    """Logistic regression"""
    def __init__(self, num_feature, output_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(num_feature, output_size)

    def forward(self, x):
        return self.linear(x)

    
class MLP(nn.Module):
    """Neural Networks"""
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1000),
            nn.Tanh(),

            nn.Linear(1000, output_dim))

    def forward(self, x):
        return self.model(x)
    
    
class three_layer_MLP(nn.Module):
    """Neural Networks"""
    def __init__(self, input_dim, output_dim):
        super(three_layer_MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 600),
            nn.Dropout(0.2),
            nn.ReLU(),
            
            nn.Linear(600, 300),
            nn.Dropout(0.2),
            nn.ReLU(),
            
            nn.Linear(300, 100),
            nn.Dropout(0.2),
            nn.ReLU(),
            
            nn.Linear(100, output_dim))

    def forward(self, x):
        return self.model(x)
    

class MnistCNN_(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MnistCNN_, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2))
        
        self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    
