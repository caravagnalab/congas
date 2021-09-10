import pyro.distributions as dist
import torch

from pyro import poutine
from pyro.infer.autoguide import AutoDelta
import numpy as np
from congas.models.Model import Model
from congas.utils import log_sum_exp, entropy_mixture, entropy_per_segment
import congas.building_blocks as lks

from sklearn.cluster import KMeans
import torch.distributions as tdist



class LatentCategorical(Model):
    params = {'K': 2, 'probs': torch.tensor([0.2, 0.3, 0.3, 0.1, 0.1]), 'hidden_dim': 5, 'a' : 1, 'b' : 100,
              'theta_shape_rna': None, 'theta_rate_rna': None,'theta_shape_atac': None, 'theta_rate_atac': None,
              'batch_size': None, "init_probs" : 5, 'norm_init_sd_rna' : None, "norm_init_sd_atac" : None,
              'mixture': None, "nb_size_init_atac": None,"nb_size_init_rna": None, "binom_prior_limits" : [10,10000],
              "likelihood_rna" : "NB", "likelihood_atac" : "NB", 'lambda' : 1, "latent_type" : "D", "Temperature" : 1/100, "equal_sizes_sd" : True}

    data_name = set(['data_rna', 'data_atac', 'pld', 'segments', 'norm_factor_rna', 'norm_factor_atac'])

    def __init__(self, data_dict):
        self._params = self.params.copy()
        self._data = None
        super().__init__(data_dict, self.data_name)

    def model(self,i = 1,  *args, **kwargs):
        pass

    def guide:
        pass

