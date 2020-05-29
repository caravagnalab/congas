import pyro
import pyro.distributions as dist
import torch
from anneal.models.Model import Model
from pyro.ops.indexing import Vindex
from pyro import poutine
from pyro.infer.autoguide import AutoDelta
from torch.distributions import constraints



class HmmMixtureRNA(Model):

# Same as before but it assumes a structure of dependence between two adjacent segments,
# probably works very well with dyploid corrected scRNA-seq data changing t

# Currently it is basically the same model as below
# TODO: add point gamma prior on theta (maybe test if necessary)

    params = {'K': 2, 'cnv_mean': 2, 'probs': torch.tensor([0.1, 0.1, 0.2, 0.3, 0.2, 0.1]), 'hidden_dim': 6,
                  'theta_scale': 9, 'theta_rate': 3, 'batch_size': None,
                  'mixture': None, 'gamma_multiplier' : 4, 't':  0.1}
    data_name = set(['data', 'mu', 'pld', 'segments'])

    def __init__(self, data_dict):
        self.params['mixture'] = 1 / torch.ones(self.params['K'])
        self._params = self.params
        self._data = None
        super().__init__(data_dict, self.data_name)

    def model(self, *args, **kwargs):
        I, N = self._data['data'].shape
        batch = N if self._params['batch_size'] else self._params['batch_size']
        weights = pyro.sample('mixture_weights', dist.Dirichlet(torch.ones(self._params['K'])))

        with pyro.plate('components', self._params['K']):
            probs_z = pyro.sample("cnv_probs",
                                  dist.Dirichlet(self._params['t'] * torch.eye(self._params['hidden_dim']) + (1- self._params['t']))
                                  .to_event(1))

        with pyro.plate("data2", N, batch):
            theta = pyro.sample('norm_factor', dist.Gamma(self._params['theta_scale'], self._params['theta_rate']))

        with pyro.plate('data', N, batch):
            z = 0
            assignment = pyro.sample('assignment', dist.Categorical(weights), infer={"enumerate": "parallel"})
            for i in pyro.markov(range(I)):
                z = pyro.sample("z_{}".format(i), dist.Categorical(Vindex(probs_z)[assignment, z]),
                                infer={"enumerate": "parallel"})
                pyro.sample('obs_{}'.format(i), dist.Poisson((z * theta * self._data['mu'][i])
                                                             + 1e-8), obs=self._data['data'][i, :])

    def guide(self,MAP = False,*args, **kwargs):
        if (MAP):
            return AutoDelta(poutine.block(self.model, expose=['mixture_weights', 'norm_factor', 'cnv_probs']),
                             init_loc_fn=self.init_fn())
        else:
            def guide_ret(*args, **kwargs):
                I, N = self._data['data'].shape
                batch = N if self._params['batch_size'] else self._params['batch_size']

                param_weights = pyro.param("param_weights", lambda: torch.ones(self._params['K']) / self._params['K'],
                                           constraint=constraints.simplex)
                cnv_mean = pyro.param("param_cnv_mean", lambda: self.create_gaussian_init_values(),
                                      constraint=constraints.positive)
                cnv_var = pyro.param("param_cnv_var", lambda: torch.ones(1) * self._params['cnv_var'],
                                     constraint=constraints.positive)
                gamma_scale = pyro.param("param_gamma_scale", lambda: torch.mean(
                    self._data['data'] / (2 * self._data['mu'].reshape(self._data['data'].shape[0], 1)), axis=0) *
                                                                      self._params['gamma_multiplier'],
                                         constraint=constraints.positive)
                gamma_rate = pyro.param("param_rate", lambda: torch.ones(1) * self._params['gamma_multiplier'],
                                        constraint=constraints.positive)
                pyro.sample('mixture_weights', dist.Dirichlet(param_weights))

                with pyro.plate('segments', I):
                    with pyro.plate('components', self._params['K']):
                        pyro.sample('cnv_probs', dist.LogNormal(torch.log(cnv_mean), cnv_var))

                with pyro.plate("data2", N, batch):
                    pyro.sample('norm_factor', dist.Gamma(gamma_scale, gamma_rate))

            return guide_ret

    def init_fn(self):
        def init_function(site):
            I, N = self._data['data'].shape
            if site["name"] == "cnv_probs":
                return (self._params['t'] * torch.eye(self._params['hidden_dim']) + (1- self._params['t'])).repeat([self._params['K'],1,1])
            if site["name"] == "mixture_weights":
                return self._params['mixture']
            if site["name"] == "norm_factor":
                return torch.mean(self._data['data'] / (2 * self._data['mu'].reshape(self._data['data'].shape[0],1)), axis=0)
            raise ValueError(site["name"])
        return init_function



