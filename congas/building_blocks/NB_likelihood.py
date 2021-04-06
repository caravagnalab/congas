import pyro.distributions as dist
import torch
from congas.utils import log_sum_exp

def NB_likelihood_aux(x, segment_fact_marg, size, mod = "rna"):

    I, N = x._data['data_{}'.format(mod)].shape

    K = x._params['K']

    mean = (segment_fact_marg.reshape([K, I, 1]) * x._data['norm_factor_{}'.format(mod)].reshape([1, N]))  # / norm_f

    lk = dist.NegativeBinomial(probs=mean / (mean + size.reshape([I, 1])),
                               total_count=size.reshape([I, 1]).repeat([1, N])).log_prob(
        x._data['data_{}'.format(mod)].repeat([K, 1, 1]))

    return lk

def NB_likelihood_aux2(x,segment_fact, cna_probs, cat_vector, size, mod = "rna"):

    I, N = x._data['data_{}'.format(mod)].shape

    K = x._params['K']

    H = x._params['hidden_dim']

    mean = (segment_fact.reshape([1, 1, I, 1]) * cat_vector.reshape([H,1,1,1]) * x._data['norm_factor_{}'.format(mod)].reshape([1,1,1, N]))  # / norm_f


    lk = dist.NegativeBinomial(probs=mean / (mean + size.reshape([1,1,I, 1])),
                               total_count=size.reshape([I, 1]).repeat([1,1,1, N])).log_prob(
        x._data['data_{}'.format(mod)].repeat([H,K, 1, 1]))

    lk += torch.log(cna_probs.reshape([H,K,I,1]))

    return log_sum_exp(lk)



def NB_likelihood(x, segment_fact_marg, weights, size, mod = "rna"):
    lk = NB_likelihood_aux(x, segment_fact_marg, size, mod)
    lk = lk + torch.log(weights).reshape([x._params['K'], 1, 1])
    summed_lk = log_sum_exp(lk)
    return summed_lk.sum()


def NB_likelihood2(x, segment_fact, cna_probs, cat_vector,weights, size, mod = "rna"):
    lk = NB_likelihood_aux2(x, segment_fact, cna_probs, cat_vector, size, mod )
    lk = lk + torch.log(weights).reshape([x._params['K'], 1, 1])
    summed_lk = log_sum_exp(lk)
    return summed_lk.sum()