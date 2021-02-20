# Federated Learning

This is a simple implementation of **federated learning (FL)** with the **noising before model aggregation FL (NbAFL)** strategy. The bare FL model (without NbAFL) is actually the reproduction of the paper [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629) and the NbAFL algorithm was proposed in [Federated Learning with Differential Privacy: Algorithms and Performance Analysis](https://arxiv.org/abs/1911.00222)


## Requirements
- PyTorch
- NumPy

## Files
- FLModel.py: definition of the FL client and FL server class.
- MLModel.py: CNN model for MNIST and FEMNIST datasets
- DPMechanisms.py: generate gaussian noise
- utils.py: sample MNIST in a non-i.i.d. manner

## Usag
1. Download MNIST dataset
2. Set parameters in test.py/test.ipynb
3. Run ```python test.py``` or Execute test.ipynb to train model on MNIST dataset

### FL model parameters
```python
# code segment in test.py/test.ipynb
lr = 0.001
fl_param = {
    'output_size': 10,          # number of units in output layer
    'client_num': client_num,   # number of clients
    'model': MnistCNN,  # model (use FEMnistCNN for FEMNIST dataset)
    'data': d,
    'lr': lr,           # learning rate
    'E': 5,             # number of local iterations
    'sigma': 0.5,       # noise level
    'clip': 4,          # clipping norm
    'batch_size': 128,  # number of samples per-batch
    'device': device
}
```

## Reference
[1] McMahan, Brendan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-Efficient Learning of Deep Networks from Decentralized Data. In *Proc. Artificial Intelligence and Statistics (AISTATS)*, 2017.

[2] K. Wei, J. Li, M. Ding, C. Ma, H. H. Yang, F. Farokhi, S. Jin, T. Q. S. Quek, H. V. Poor, Federated Learning with Differential Privacy: Algorithms and Performance Analysis. In *IEEE Transactions on Information Forensics & Security*, 15, pp. 3454-3469, 2020.
