# Implementation of typical DP/LDP Mechanisms
import torch
import numpy as np
import random


def clip_grad(grad, clip):
    g_shape = grad.shape
    grad.flatten()
    grad = grad / np.max((1, float(torch.norm(grad, p=2)) / clip))
    grad.view(g_shape)
    return grad


def gaussian_noise(data_shape, s, sigma, device=None):
    """
    Gaussian noise for CDP-FedAVG-LS Algorithm
    """
    return torch.normal(0, sigma * s, data_shape).to(device)


def gaussian_noise_mask(grad, s, epsilon, delta, mask, device):
    """
    add noise on rated items
    """
    noised_g = gaussian_noise(grad, s, epsilon, delta, device) * mask
    return noised_g


def laplace_noise(grad, s, epsilon, mask):
    """
    generate Laplace Noise: add noise on rated items
    """
    noise = np.random.laplace(0, s/epsilon, grad.shape) * mask
    return grad + noise


def naive_ldp(mask, p):
    """
    perform simple LDP to protect "which items are rated by a user"
    y: mask of rating matrix, y = (r > 0)
            0,       with probability p/2,
    y* =    1,       with probability p/2,
            y_{ij},  with probability 1 - p
    """
    # without LDP mechanism
    if p == 0:
        return mask[0]
    y = mask[0]
    y_star = y
    for i in range(y_star.shape[0]):
        rnd = random.random()
        if rnd < p/2:
            y_star[i] = 0
        elif rnd < p:
            y_star[i] = 1
    return y_star

