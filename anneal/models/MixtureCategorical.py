import pyro
import pyro.distributions as dist
import numpy as np
import torch
from pyro.infer import config_enumerate

from models.Model import Model
from pyro.ops.indexing import Vindex
from pyro import poutine
from pyro.infer.autoguide import AutoDelta
from torch.distributions import constraints






class MixtureCategorical(Model):

    params = {'K': 2, 'cnv_mean': 2, 'probs': torch.tensor([0.1, 0.1, 0.2, 0.3, 0.2, 0.1]), 'hidden_dim': 6,
              'theta_scale': 3, 'theta_rate': 1, 'batch_size': None,
              'mixture': None}
    data_name = set(['data', 'mu', 'pld', 'segments'])

    def __init__(self, param_dict):
        if param_dict['probs'] is None:
            param_dict['probs'] = torch.ones(param_dict['hidden_dim']) * 1/param_dict['hidden_dim']
        super().__init__(param_dict, self.params_name)

    def model(self,*args, **kwargs):
        I, N = self.params['data'].shape
        weights = pyro.sample('mixture_weights', dist.Dirichlet((1 / self.params['K']) * torch.ones(self.params['K'])))
        with pyro.plate('segments', I):
            with pyro.plate('components', self.params['K']):
                cnv_probs = pyro.sample("cnv_probs", dist.Dirichlet(
                    self.params['probs'] * 1 / torch.ones(self.params['hidden_dim'])))
                cc = pyro.sample("cc", dist.Categorical(cnv_probs), infer={"enumerate": "sequential"})
        with pyro.plate('data', N, self.params['batch_size']):
            assignment = pyro.sample('assignment', dist.Categorical(weights), infer={"enumerate": "parallel"})
            theta = pyro.sample('norm_factor', dist.Gamma(self.params['theta_scale'], self.params['theta_rate']))
            for i in pyro.plate('segments2', I):
                pyro.sample('obs_{}'.format(i), dist.Poisson((Vindex(cc)[assignment,i] * theta * self.params['mu'][i])
                                                             + 1e-8), obs=self.params['data'][i, :])
    def guide(self, *args, **kwargs):
        def guide_ret(*args, **kargs):
            I, N = self.params['data'].shape
            param_weights = pyro.param("param_weights", lambda: torch.ones(self.params['K']) / self.params['K'],
                               constraint=constraints.simplex)
            cnv_weights = pyro.param("cnv_weights", lambda: torch.ones((3,7,5)),
                               constraint=constraints.simplex)
            gamma = pyro.param("param_gamma", lambda: torch.ones(N) * 2,
                               constraint=constraints.positive)
            weights = pyro.sample('mixture_weights', dist.Delta(param_weights, event_dim = 1))

            with pyro.plate('segments', I):
                with pyro.plate('components', self.params['K']):
                    cnv_probs = pyro.sample("cnv_probs", dist.Delta(cnv_weights, event_dim=1))

                    pyro.sample("cc", dist.Categorical(cnv_probs))
            with pyro.plate('data', N, self.params['batch_size']):
                pyro.sample('assignment', dist.Categorical(weights), infer={"enumerate": "parallel"})
                pyro.sample('norm_factor', dist.Delta(gamma))
        return guide_ret


    def final_guide(self, MAPs):
        def full_guide(self, *args, **kargs):
            I,N = self.params['data'].shape
            with poutine.block(hide=["cnv_probs","norm_factor", "mixture_weights"]):
                weights = pyro.sample('mixture_weights',
                                      dist.Dirichlet(MAPs['mixture_weights'].detach()))
                with pyro.plate('segments', I):
                    with pyro.plate('components', self.params['K']):
                        cnv_probs = pyro.sample("cnv_probs", dist.Dirichlet(
                            MAPs['cnv_probs'].detach()))
                with pyro.plate('data', N, self.params['batch_size']):
                    assignment_probs = pyro.param('assignment_probs',
                                                  torch.ones(N, self.params['K']) / self.params['K']
                                                  , constraint=constraints.unit_interval)
                    assignment = pyro.sample('assignment', dist.Categorical(assignment_probs), infer={"enumerate": "parallel"})
                    pyro.sample('norm_factor',
                                        dist.Delta(MAPs['norm_factor'].detach()))
                    for i in pyro.plate('segments2', I):
                        pyro.sample('copy_number_{}'.format(i),
                                         dist.Categorical(Vindex(cnv_probs)[assignment, i, :]),
                                         infer={"enumerate": "parallel"})

        return full_guide

    def create_dirichlet_init_values(self):

        bins = self.params['hidden_dim'] * 2
        low_prob = 1 / bins
        high_prob = low_prob * (self.params['hidden_dim'] + 1)
        init = torch.zeros(self.params['K'], self.params['segments'], self.params['hidden_dim'])

        for i in range(len(self.params['pld'])):
            for j in range(self.params['hidden_dim']):
                for k in range(self.params['K']):
                    if k == 0:
                        init[k, i, j] = high_prob if j == torch.ceil(self.params['pld'][i]) else low_prob
                    else:
                        init[k, i, j] = high_prob if j == torch.floor(self.params['pld'][i]) else low_prob

        return init


