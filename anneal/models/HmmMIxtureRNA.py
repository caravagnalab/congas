import pyro
import pyro.distributions as dist
import numpy as np
import torch
from models.Model import Model
from pyro.ops.indexing import Vindex
from pyro import poutine
from pyro.infer.autoguide import AutoDelta


class HmmMixtureRNA(Model):

    def __init__(self):
        pass
# Same as before but it assumes a structure of dependence between two adjacent segments,
# probably works very well with dyploid corrected scRNA-seq data changing t

# Currently it is basically the same model as below
# TODO: add point gamma prior on theta (maybe test if necessary)


    def model(data, mu, K=2, hidden_dim=6, batch_size=None, theta_prior_mean=np.log(2.1),
                   theta_prior_scale=0.1, t = 0.3):
        I, N = data.shape
        weights = pyro.sample('mixture_weights', dist.Dirichlet((1 / K) * torch.ones(K)))
        with pyro.plate('components', K):
            # Note here we give less probability to a transition in the same state
            cnv_probs = pyro.sample("cnv_probs",
                                    dist.Dirichlet(t * torch.eye(hidden_dim) +(1-t))
                                    .to_event(1))
        with pyro.plate("data", N, batch_size):
            x = 0
            assignment = pyro.sample('assignment', dist.Categorical(weights), infer={"enumerate": "parallel"})
            theta = pyro.sample('norm_factor', dist.LogNormal(theta_prior_mean, theta_prior_scale))
            for t in pyro.markov(range(I)):
                x = pyro.sample('copy_number_{}'.format(t), dist.Categorical(Vindex(cnv_probs)[assignment, x]),
                                infer={"enumerate": "parallel"})
                pyro.sample('obs_{}'.format(t), dist.Poisson((x * theta * mu[t]) + 1e-8), obs=data[t, :])

    def guide(self):
        pass

    def init_fn(self):
        pass

    def write_results(self, prefix):
        pass

