import pyro
import pyro.distributions as dist
import torch

from congas.models.Model import Model
from pyro.ops.indexing import Vindex
from pyro import poutine
from torch.distributions import constraints






class MixtureCategorical(Model):

    params = {'K': 2, 'cnv_mean': 2, 'probs': torch.tensor([0.1, 0.1, 0.2, 0.3, 0.2, 0.1]), 'hidden_dim': 6,
              'theta_scale': 9, 'theta_rate': 3, 'batch_size': None,
              'mixture': torch.tensor([1,1]), 'gamma_multiplier' : 4}
    data_name = set(['data', 'mu', 'pld', 'segments'])

    def __init__(self, data_dict):
        self._params = self.params.copy()
        self._data = None
        super().__init__(data_dict, self.data_name)

    def model(self,*args, **kwargs):
        I, N = self._data['data'].shape
        batch = N if self._params['batch_size'] else self._params['batch_size']
        weights = pyro.sample('mixture_weights', dist.Dirichlet((1 / self._params['K']) * torch.ones(self._params['K'])))

        with pyro.plate('segments', I):
            with pyro.plate('components', self._params['K']):
                cc = pyro.sample("cc", dist.Categorical(self.create_dirichlet_init_values()))

        with pyro.plate("data2", N, batch):
            theta = pyro.sample('norm_factor', dist.Gamma(self._params['theta_scale'], self._params['theta_rate']))

        with pyro.plate('data', N, batch):
            assignment = pyro.sample('assignment', dist.Categorical(weights), infer={"enumerate": "parallel"})
            for i in pyro.plate('segments2', I):
                pyro.sample('obs_{}'.format(i), dist.Poisson((Vindex(cc)[...,assignment,i] * theta * self._data['mu'][i])
                                                             + 1e-8), obs=self._data['data'][i, :])
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

    def full_guide(self, MAP=False, *args, **kwargs):
        def full_guide_ret(*args, **kargs):
            I, N = self._data['data'].shape
            batch = N if self._params['batch_size'] else self._params['batch_size']

            with poutine.block(hide_types=["param"]):  # Keep our learned values of global parameters.
                self.guide(MAP)()
            with pyro.plate('data', N, batch):
                assignment_probs = pyro.param('assignment_probs', torch.ones(N, self._params['K']) / self._params['K'],
                                              constraint=constraints.unit_interval)
                pyro.sample('assignment', dist.Categorical(assignment_probs), infer={"enumerate": "parallel"})
        return full_guide_ret

    def create_dirichlet_init_values(self):

        bins = self._params['hidden_dim'] * 120
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


