# Federated Learning with Differential Privacy

This repo implements **Federated Learning (FL)** with **Differential Privacy (DP)** using the FedAvg algorithm from [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629). Each client performs local DP-SGD updates and perturbs model parameters with Gaussian noise. The noise multiplier is calibrated via Rényi DP analysis (see `rdp_analysis.py`) using tighter bounds from [3-5].

## Requirements
- torch, torchvision
- numpy
- scipy
- kymatio (only required for the ScatterNet baseline in `MLModel.py`)

## Repository layout
- `FLModel.py`: Federated client/server classes, FedAvg aggregation, and DP-SGD update loop.
- `MLModel.py`: Model definitions (MNIST CNN, ScatterNet linear head, MLP baselines).
- `rdp_analysis.py`: Rényi DP accounting and Gaussian noise calibration utilities.
- `utils.py`: MNIST non-i.i.d. data partitioning and Gaussian noise helper.
- `test_cnn.ipynb`: Example notebook for the CNN baseline.
- `test_scatter_linear.ipynb`: Example notebook for ScatterNet features + linear head.

## Usage
Open and run `test_cnn.ipynb` (or `test_scatter_linear.ipynb`) in Jupyter to reproduce experiments.

### FL configuration
Below is a representative parameter dict used by `FLServer`/`FLClient` (keys must match what `FLModel.py` expects):

```python
# code segment in test_cnn.ipynb
lr = 0.1
fl_param = {
    'output_size': 10,          # number of units in output layer
    'client_num': client_num,   # number of clients
    'model': MNIST_CNN,         # model class or 'scatter'
    'data': d,                  # list of client datasets + test set
    'lr': lr,                   # learning rate
    'E': 500,                   # number of local iterations
    'batch_size': 32,           # local batch size
    'eps': 4.0,                 # privacy budget (epsilon)
    'delta': 1e-5,              # DP delta
    'q': 0.01,                  # Poisson sampling rate
    'clip': 0.2,                # clipping norm
    'tot_T': 10,                # number of communication rounds
    'device': 'cuda',           # training device
}
```

### Notes
- `FLServer` calibrates the noise scale with `calibrating_sampled_gaussian` using `E * tot_T` as the number of compositions.
- If you set `model` to `'scatter'`, `MLModel.ScatterLinear` is used with the Scattering2D front-end.

## References
[1] McMahan, Brendan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-Efficient Learning of Deep Networks from Decentralized Data. In *AISTATS*, 2017.

[2] Abadi, Martin, et al. Deep learning with differential privacy. In *CCS*. 2016.

[3] Mironov, Ilya, Kunal Talwar, and Li Zhang. R\'enyi differential privacy of the sampled gaussian mechanism. arXiv preprint 2019.

[4] Canonne, Clément L., Gautam Kamath, and Thomas Steinke. The discrete gaussian for differential privacy. In *NeurIPS*, 2020.

[5] Asoodeh, S., Liao, J., Calmon, F.P., Kosut, O. and Sankar, L., A better bound gives a hundred rounds: Enhanced privacy guarantees via f-divergences. In *ISIT*, 2020.
