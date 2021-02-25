import pyro
import pyro.distributions as dist
import torch

import numpy as np
from congas.models.Model import Model
from pyro.ops.indexing import Vindex
from pyro import poutine
from torch.distributions import constraints






class MixtureCategorical(Model):

    params = {'K': 2, 'cnv_mean': 2, 'probs': torch.tensor([0.1, 0.1, 0.2, 0.3, 0.2, 0.1]), 'hidden_dim': 6,
              'theta_scale': None, 'theta_rate': None, 'batch_size': None,
              'mixture': torch.tensor([1,1]), 'gamma_multiplier' : 4, }
    data_name = set(['data', 'mu', 'pld', 'segments', 'theta'])

    def __init__(self, data_dict):
        self._params = self.params.copy()
        self._data = None
        super().__init__(data_dict, self.data_name)

    def model(self,*args, **kwargs):

        I, N = self._data['data'].shape

        batch = N if self._params['batch_size'] else self._params['batch_size']

        weights = pyro.sample('mixture_weights', dist.Dirichlet((1 / self._params['K']) * torch.ones(self._params['K'])))
        cat_vector = torch.tensor(np.arange(self._params['hidden_dim']) + 1, dtype = torch.float)

        with pyro.plate('segments', I):
            segment_factor = pyro.sample('segment_factor', dist.Gamma(self._params['theta_scale'], self._params['theta_rate']))
            with pyro.plate('components', self._params['K']):
                cc = pyro.sample("CNV_probabilities", dist.Dirichlet(self.create_dirichlet_init_values()))

        with pyro.plate('data', N, batch):

                # p(x|z_i) = Poisson(marg(cc * theta * segment_factor))

                segment_fact_cat = torch.matmul(segment_factor.reshape([I,1]) , cat_vector.reshape([1, self._params['hidden_dim']]))
                segment_fact_marg = segment_fact_cat * cc
                segment_fact_marg = torch.sum(segment_fact_marg, dim = -1)

                # p(z_i| D, X ) = lk(z_i) * p(z_i | X) / sum_z_i(lk(z_i) * p(z_i | X))
                # log(p(z_i| D, X )) = log(lk(z_i)) + log(p(z_i | X)) - log_sum_exp(log(lk(z_i)) + log(p(z_i | X)))

                pyro.factor("lk", self.likelihood(segment_fact_marg, weights, self._params['theta']))

    def guide(self,MAP = False,  *args, **kwargs):
        def guide_ret(*args, **kargs):

            I, N = self._data['data'].shape
            batch = N if self._params['batch_size'] else self._params['batch_size']

            if(MAP):
                gamma_MAP = pyro.param("param_gamma_MAP", lambda : torch.mean(self._data['data'] / (
                    2 * self._data['mu'].reshape(self._data['data'].shape[0], 1)), axis=0),
                    constraint=constraints.positive)
            else:
                gamma_scale = pyro.param("param_gamma_scale", lambda: torch.mean(
                    self._data['data'] / (2 * self._data['mu'].reshape(self._data['data'].shape[0], 1)), axis=0) *
                                                                      self._params['gamma_multiplier'],
                                         constraint=constraints.positive)
                gamma_rate = pyro.param("param_rate", lambda: torch.ones(1) * self._params['gamma_multiplier'],
                                        constraint=constraints.positive)
            param_weights = pyro.param("param_weights", lambda: torch.ones(self._params['K']) / self._params['K'],
                                       constraint=constraints.simplex)
            cnv_weights = pyro.param("param_cnv_weights", lambda:  self.create_dirichlet_init_values(),
                               constraint=constraints.simplex)


            weights = pyro.sample('mixture_weights', dist.Dirichlet(param_weights))

            with pyro.plate('segments', I):
                with pyro.plate('components', self.params['K']):
                    pyro.sample("cc", dist.Categorical(cnv_weights))

            with pyro.plate("data2", N, batch):
                if(MAP):
                    pyro.sample('norm_factor', dist.Delta(gamma_MAP))
                else:
                    pyro.sample('norm_factor', dist.Gamma(gamma_scale, gamma_rate))

        return guide_ret


    def create_dirichlet_init_values(self):

        bins = self._params['hidden_dim'] * 10
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


    def likelihood(self, ):







