import pyro
import pyro.distributions as dist
import torch

from pyro import poutine
from pyro.infer.autoguide import AutoDelta
import numpy as np
from congas.models.Model import Model
from torch.distributions import constraints
from congas.utils import log_sum_exp, entropy


class MixtureCategorical(Model):
    params = {'K': 2, 'cnv_mean': 2, 'probs': torch.tensor([0.2, 0.3, 0.3, 0.1, 0.1]), 'hidden_dim': 5,
              'theta_scale': None, 'theta_rate': None, 'batch_size': None,
              'mixture': None, 'gamma_multiplier': 4, "nb_size_init": None, "binom_prior_limits" : [0.01,10000]}
    data_name = set(['data', 'mu', 'pld', 'segments', 'norm_factor'])

    def __init__(self, data_dict):
        self._params = self.params.copy()
        self._data = None
        super().__init__(data_dict, self.data_name)

    def model(self, *args, **kwargs):

        I, N = self._data['data'].shape

        batch = N if self._params['batch_size'] else self._params['batch_size']


        weights = pyro.sample('mixture_weights',
                              dist.Dirichlet((1. / self._params['K']) * torch.ones(self._params['K'])))
        cat_vector = torch.tensor(np.arange(self._params['hidden_dim']) + 1, dtype=torch.float)

        with pyro.plate('segments', I):

            sizes = pyro.sample("NB_size", dist.Uniform(self._params['binom_prior_limits'][0], self._params['binom_prior_limits'][1]))
            segment_factor = pyro.sample('segment_factor',
                                         dist.Gamma(self._params['theta_scale'], self._params['theta_rate']))
            with pyro.plate('components', self._params['K']):
                cc = pyro.sample("CNV_probabilities", dist.Dirichlet(self._params['probs']))

        with pyro.plate('data', N, batch):
            # p(x|z_i) = Poisson(marg(cc * theta * segment_factor))

            segment_fact_cat = torch.matmul(segment_factor.reshape([I, 1]),
                                            cat_vector.reshape([1, self._params['hidden_dim']]))
            segment_fact_marg = segment_fact_cat * cc
            segment_fact_marg = torch.sum(segment_fact_marg, dim=-1)

            # p(z_i| D, X ) = lk(z_i) * p(z_i | X) / sum_z_i(lk(z_i) * p(z_i | X))
            # log(p(z_i| D, X )) = log(lk(z_i)) + log(p(z_i | X)) - log_sum_exp(log(lk(z_i)) + log(p(z_i | X)))

            pyro.factor("lk", self.final_likelihood(segment_fact_marg, weights, sizes) - entropy(cc))

    def guide(self, *args, **kwargs):
        return AutoDelta(poutine.block(self.model, expose=["NB_size", "mixture_weights",
                                                           "segment_factor", "CNV_probabilities"]),
                         init_loc_fn=self.init_fn())

    def init_fn(self):

        def init_function(site):
            if site["name"] == "CNV_probabilities":
                return self.create_dirichlet_init_values()
            if site["name"] == "mixture_weights":
                return self._params['mixture']
            if site["name"] == "NB_size":
                return self._params['nb_size_init']
            if site["name"] == "segment_factor":
                return dist.Gamma(self._params['theta_scale'], self._params['theta_rate']).mean
            raise ValueError(site["name"])

        return init_function

    def create_dirichlet_init_values(self):

        bins = self._params['hidden_dim'] * 6
        low_prob = 1 / bins
        high_prob = 1 - (low_prob * (self._params['hidden_dim'] - 1))

        init = torch.zeros(self._params['K'], self._data['segments'], self._params['hidden_dim'])

        for i in range(len(self._data['pld'])):
            for j in range(self._params['hidden_dim']):
                for k in range(self._params['K']):
                    if k == 0:
                        init[k, i, j] = high_prob if j == torch.ceil(self._data['pld'][i]) else low_prob
                    else:
                        init[k, i, j] = high_prob if j == torch.floor(self._data['pld'][i]) else low_prob

        return init

    def likelihood_model(self, segment_fact_marg, weights, size):
        lk = torch.zeros(self._params['K'], self._data['data'].shape[1], self._data['data'].shape[0])
        if self._params['K'] == 1:
            #norm_f = torch.sum(segment_fact_marg)
            for i in range(self._data['data'].shape[0]):
                mean = (segment_fact_marg[0, i] * self._data['norm_factor']) #/ norm_f
                lk[0, :, i] = torch.log(weights) + dist.NegativeBinomial(probs = size[i] / (mean + size[i]), total_count = size[i]).log_prob(
                    self._data['data'][i, :])
            return lk

        for k in range(self._params['K']):
            #norm_f = torch.sum(segment_fact_marg[k,:])
            for i in range(self._data['data'].shape[0]):
                mean = (segment_fact_marg[k, i] * self._data['norm_factor'])# / norm_f
                lk[k, :, i] = dist.NegativeBinomial(probs = mean / (mean + size[i]), total_count = size[i]).log_prob(
                    self._data['data'][i, :])

        return lk

    def final_likelihood(self, segment_fact_marg, weights, size):

        lk = self.likelihood_model(segment_fact_marg, weights, size)
        lk = torch.sum(lk, dim=2) + torch.log(weights).reshape([self._params['K'], 1])
        summed_lk = log_sum_exp(lk)
        return summed_lk.sum()

    def likelihood(self, inf_params):
        I, N = self._data['data'].shape
        cat_vector = torch.tensor(np.arange(self._params['hidden_dim']) + 1, dtype=torch.float)
        segment_fact = torch.matmul(inf_params["segment_factor"].reshape([I, 1]),
                                    cat_vector.reshape([1, self._params['hidden_dim']]))

        segment_fact_marg = segment_fact * inf_params["CNV_probabilities"]
        segment_fact_marg = torch.sum(segment_fact_marg, dim=-1)

        lk = self.likelihood_model(segment_fact_marg, inf_params["mixture_weights"], inf_params["NB_size"])
        return lk

    def calculate_cluster_assignements(self, inf_params):

        lk = self.likelihood(inf_params)
        lk = torch.sum(lk, dim=2) + torch.log(inf_params["mixture_weights"]).reshape([self._params['K'], 1])

        summed_lk = log_sum_exp(lk)
        ret = lk - summed_lk
        res = {"assignment_probs": torch.exp(ret).detach().numpy()}
        return res
