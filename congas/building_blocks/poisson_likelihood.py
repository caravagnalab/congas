import pyro.distributions as dist
import torch
from congas.utils import log_sum_exp

def poisson_likelihood_aux(x, segment_fact_marg, mod = "rna"):

    I, N = x._data['data_{}'.format(mod)].shape

    K = x._params['K']

    mean = (segment_fact_marg.reshape([K, I, 1]) * x._data['norm_factor_{}'.format(mod)].reshape([1, N]))  # / norm_f

    lk = dist.Poisson(mean).log_prob(
        x._data['data_{}'.format(mod)])


    return lk


def poisson_likelihood(x, segment_fact_marg, weights, mod = "rna"):
    lk = poisson_likelihood_aux(x, segment_fact_marg, mod)
    lk =  lk.sum(dim = 1) + torch.log(weights).reshape([x._params['K'], 1])
    summed_lk = log_sum_exp(lk)
    return summed_lk.sum()

