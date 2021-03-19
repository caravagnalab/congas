import pyro
import pyro.distributions as dist
import torch
from congas.models.Model import Model
from pyro.ops.indexing import Vindex
from pyro import poutine
from pyro.infer.autoguide import AutoDelta
from torch.distributions import constraints



class HmmMixtureRNA(Model):
    """

    It is basically :class:`~congas.models.MixtureDirichlet.MixtureDirichlet` but with markov property


    Model parameters:
        T = max number of clusters (default = 6)
        hidden_dim = hidden dimensions (should be len(probs))
        theta_scale = scale for the normalization factor variable (default = 3)
        theta_rate = rate for the normalization factor variable (default = 1)
        batch_size = batch size (default = None)
        mixture = prior for the mixture weights (default = 1/torch.ones(K))
        gamma_multiplier = multiplier Gamma(rate * gamma_multiplier, shape  * gamma_multiplier) when we also want to
        infer the shape and rate parameter (i.e. when MAP = FALSE) (default = 4)
        t = probability of remaining in the same state (default=0.1)






    TODO:
        test on meaningful datasets

    """

    params = {'K': 2, 'hidden_dim': 6,
                  'theta_scale': 9, 'theta_rate': 3, 'batch_size': None,
                  'mixture': torch.tensor([1.,1.]), 'gamma_multiplier' : 4, 't':  0.1}
    data_name = set(['data', 'mu', 'pld', 'segments'])

    def __init__(self, data_dict):
        self._params = self.params.copy()
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

    def guide(self,*args, **kwargs):
        return AutoDelta(poutine.block(self.model, expose=['mixture_weights', 'norm_factor', 'cnv_probs', 'pi']),
                         init_loc_fn=self.init_fn())

    def full_guide(self, *args, **kwargs):
        def full_guide_ret(*args, **kargs):
            I, N = self._data['data'].shape
            batch = N if self._params['batch_size'] else self._params['batch_size']

            with poutine.block(hide_types=["param"]):  # Keep our learned values of global parameters.
                self.guide()()
            with pyro.plate('data', N, batch):
                assignment_probs = pyro.param('assignment_probs', torch.ones(N, self._params['K']) / self._params['K'],
                                              constraint=constraints.unit_interval)
                pyro.sample('assignment', dist.Categorical(assignment_probs), infer={"enumerate": "parallel"})

        return full_guide_ret

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



