import numpy as np
import torch


def calc_BIC(lk, n_params, N):
    return np.log(N) * n_params - 2 * lk


def calc_AIC(lk, n_params):
    return 2 * n_params - 2 * lk


def calc_ICL(lk, n_params, N, entropy):
    bic = calc_BIC(lk, n_params, N)
    return bic - entropy


def calc_entropy(x):
    entr = torch.zeros(x.shape[1])

    for i in range(x.shape[1]):
            entr[i] = torch.sum(x[:, i] + torch.log(x[:, i] + 1e-10))
    return entr.sum()


def calculate_number_of_params(params):
    res = 0
    for i in params:
        res += np.prod(params[i].shape)
    return res
