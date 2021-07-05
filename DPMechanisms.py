# Implementation of typical DP/LDP Mechanisms
import torch
import numpy as np
import random


def gaussian_noise(data_shape, s, sigma, device=None):
    """
    Gaussian noise for CDP-FedAVG-LS Algorithm
    """
    return torch.normal(0, sigma * s, data_shape).to(device)

