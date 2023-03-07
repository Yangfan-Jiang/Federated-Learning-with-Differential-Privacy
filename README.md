# Federated Learning

This is a simple implementation of **Federated Learning (FL)** with **Differential Privacy (DP)**. The bare FL model (without DP) is the reproduction of the paper [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629). Each client train local model using DP-SGD ([2], [tensorflow-privacy]( https://github.com/tensorflow/privacy)) to perturb model parameters.

TODO:
- [ ] rdp_analysis.py: RDP for subsampled Gaussian [4], convert RDP to DP by Proposition 12 of [5] (tighter privacy analysis than [2, 3]).

## Requirements
- torch 1.7.1
- tensorflow-privacy 0.5.1
- numpy 1.16.2

## Files
> FLModel.py: definition of the FL client and FL server class

> MLModel.py: CNN model for MNIST datasets

> utils.py: sample MNIST in a non-i.i.d. manner

## Usag
1. Download MNIST dataset
2. Install [tensorflow-privacy]( https://github.com/tensorflow/privacy)
2. Set parameters in test.py/test.ipynb
3. Execute test.ipynb to train model on MNIST dataset

### FL model parameters
```python
# code segment in test.py/test.ipynb
lr = 0.1
fl_param = {
    'output_size': 10,          # number of units in output layer
    'client_num': client_num,   # number of clients
    'model': MnistCNN,  # model
    'data': d,          # dataset
    'lr': lr,           # learning rate
    'E': 100,           # number of local iterations
    'eps': 8.0,         # privacy budget
    'delta': 1e-5,      # approximate differential privacy: (epsilon, delta)-DP
    'q': 0.05,          # sampling rate
    'clip': 8,          # clipping norm
    'tot_T': 10,        # number of aggregation times (communication rounds)
}
```


## Reference
[1] McMahan, Brendan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-Efficient Learning of Deep Networks from Decentralized Data. In *AISTATS*, 2017.

[2] Abadi, Martin, et al. Deep learning with differential privacy. In *CCS*. 2016.

[3] TensorFlow Privacy: https://github.com/tensorflow/privacy

[4] Mironov, Ilya, Kunal Talwar, and Li Zhang. "R\'enyi differential privacy of the sampled gaussian mechanism." arXiv preprint 2019.

[5] Canonne, Clément L., Gautam Kamath, and Thomas Steinke. The discrete gaussian for differential privacy. In *NeurIPS* , 2020.