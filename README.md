# Federated Learning

This is a simple implementation of **Federated Learning (FL)** with **Differential Privacy (DP)**. The bare FL model (without DP) is actually the reproduction of the paper [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629).


## Requirements
- PyTorch
- tensorflow-privacy
- NumPy

## Files
> FLModel.py: definition of the FL client and FL server class

> MLModel.py: CNN model for MNIST and FEMNIST datasets

> DPMechanisms.py: generate gaussian noise

> utils.py: sample MNIST in a non-i.i.d. manner

## Usag
1. Download MNIST dataset
2. Set parameters in test.py/test.ipynb
3. Run ```python test.py``` or Execute test.ipynb to train model on MNIST dataset

### FL model parameters
```python
# code segment in test.py/test.ipynb
lr = 0.1
fl_param = {
    'output_size': 10,          # number of units in output layer
    'client_num': client_num,   # number of clients
    'model': MnistCNN,  # model (use FEMnistCNN for FEMNIST dataset)
    'data': d,
    'lr': lr,           # learning rate
    'E': 1,             # number of local iterations
    'eps': 4.0,         # privacy budget
    'delta': 1e-5,      # approximate differential privacy: (epsilon, delta)-DP
    'q': 0.03,          # sampling rate
    'clip': 32,         # clipping norm
    'batch_size': 128,
    'device': device
}
```

## Reference
[1] McMahan, Brendan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. "Communication-Efficient Learning of Deep Networks from Decentralized Data." In *Proc. Artificial Intelligence and Statistics (AISTATS)*, 2017.

[2] Abadi, Martin, et al. "Deep learning with differential privacy." *Proceedings of the 2016 ACM SIGSAC conference on computer and communications security*. 2016.

[3] TensorFlow Privacy: https://github.com/tensorflow/privacy

