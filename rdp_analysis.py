import numpy as np
from decimal import *
from scipy.special import comb

getcontext().prec = 128


def rdp2dp(rdp, bad_event, alpha):
    """
    convert RDP to DP, Ref:
    - Canonne, Clément L., Gautam Kamath, and Thomas Steinke. The discrete gaussian for differential privacy. In NeurIPS, 2020. (See Proposition 12)
    - Asoodeh, S., Liao, J., Calmon, F.P., Kosut, O. and Sankar, L., A better bound gives a hundred rounds: Enhanced privacy guarantees via f-divergences. In ISIT, 2020. (See Lemma 1)
    """
    return rdp + 1.0/(alpha-1) * (np.log(1.0/bad_event) + (alpha-1)*np.log(1-1.0/alpha) - np.log(alpha))


def compute_rdp(alpha, q, sigma):
    """
    RDP for subsampled Gaussian mechanism, Ref:
    - Mironov, Ilya, Kunal Talwar, and Li Zhang. R\'enyi differential privacy of the sampled gaussian mechanism. arXiv preprint 2019.
    """
    sum_ = Decimal(0.0)
    for k in range(0, alpha+1):
        sum_ += Decimal(comb(alpha, k)) * Decimal(1-q)**Decimal(alpha-k) * Decimal(q**k) * Decimal(np.e)**(Decimal(k**2-k)/Decimal(2*sigma**2))
    rdp = sum_.ln() / Decimal(alpha-1)
    return float(rdp)
 
    
def search_dp(q, sigma, bad_event, iters=1):
    """
    Given the sampling rate, variance of Gaussian noise, and privacy parameter delta, 
    this function returns the corresponding DP budget.
    """
    min_dp = 1e5
    for alpha in list(range(2, 101)):
        rdp = iters * compute_rdp(alpha, q, sigma)
        dp = rdp2dp(rdp, bad_event, alpha)
        min_dp = min(min_dp, dp)
    return min_dp
    
    
def calibrating_sampled_gaussian(q, eps, bad_event, iters=1, err=1e-3):
    """
    Calibrate noise to privacy budgets
    """
    sigma_max = 100
    sigma_min = 0.1
    
    def binary_search(left, right):
        mid = (left + right) / 2
        
        lbd = search_dp(q, mid, bad_event, iters)
        ubd = search_dp(q, left, bad_event, iters)
        
        if ubd > eps and lbd > eps:    # min noise & mid noise are too small
            left = mid
        elif ubd > eps and lbd < eps:  # mid noise is too large
            right = mid
        else:
            print("an error occurs in func: binary search!")
            return -1
        return left, right
        
    # check
    if search_dp(q, sigma_max, bad_event, iters) > eps:
        print("noise > 100")
        return -1
    
    while sigma_max-sigma_min > err:
        sigma_min, sigma_max = binary_search(sigma_min, sigma_max)
    return sigma_max