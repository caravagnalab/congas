import pyro
import pyro.distributions as dist
import torch
from congas.models.Model import Model
from pyro.ops.indexing import Vindex
from pyro import poutine
from pyro.infer.autoguide import AutoDelta
from torch.distributions import constraints
import torch.nn.functional as F

"""

Same as :class:`~congas.models.MixtureGaussian.MixtureGaussian` but the number of cluster is non-parametric
 (estimated with a stick-breaking Dirichlet Mixture Process)


Model parameters:
    T = max number of clusters (default = 6)
    cnv_var = var of the LogNorm prior (default = 0.6)
    theta_scale = scale for the normalization factor variable (default = 3)
    theta_rate = rate for the normalization factor variable (default = 1)
    batch_size = batch size (default = None)
    mixture = prior for the mixture weights (default = 1/torch.ones(K))
    gamma_multiplier = multiplier Gamma(rate * gamma_multiplier, shape  * gamma_multiplier) when we also want to 
    infer the shape and rate parameter (i.e. when MAP = FALSE) (default = 4)





TODO: 
    add support for joint inference with bulk counts (or allelic frequencies)


"""


class MixtureGaussianDMP(Model):

    params = {'T' : 6, 'cnv_mean' : 2, 'cnv_var' :0.6, 'theta_scale' : 3, 'theta_rate' : 1, 'batch_size' : None,
            'mixture' : torch.tensor([1,1]), 'alpha' : 0.0001, 'gamma_multiplier' : 5}
    data_name = set(['data', 'mu','pld', 'segments'])


    def __init__(self, data_dict):
        self._params = self.params.copy()
        self._data = None
        super().__init__(data_dict, self.data_name)

    def mix_weights(self,beta):
        beta1m_cumprod = (1 - beta).cumprod(-1)
        return F.pad(beta, (0, 1), value=1) * F.pad(beta1m_cumprod, (1, 0), value=1)

    def model(self,*args, **kwargs):
        I, N = self._data['data'].shape
        batch = N if self._params['batch_size'] else self._params['batch_size']

        with pyro.plate("beta_plate", self._params['T'] - 1):
            beta = pyro.sample("mixture_weights", dist.Beta(1, self._params['alpha']))

        with pyro.plate('segments', I):
            with pyro.plate('components', self._params['T']):
                cc = pyro.sample('cnv_probs', dist.LogNormal(torch.log(self._data['pld']), self._params['cnv_var']))

        with pyro.plate('data2', N, batch):
            theta = pyro.sample('norm_factor', dist.Gamma(self._params['theta_scale'], self._params['theta_rate']))

        with pyro.plate('data', N, batch):
            assignment = pyro.sample('assignment', dist.Categorical(self.mix_weights(beta)), infer={"enumerate": "parallel"})
            for i in pyro.plate('segments2', I):
                pyro.sample('obs_{}'.format(i), dist.Poisson((Vindex(cc)[assignment,i] * theta * self._data['mu'][i]
                                                              )
                                                             + 1e-8), obs=self._data['data'][i, :])

    def guide(self,MAP = False,*args, **kwargs):
        if (MAP):
            return AutoDelta(poutine.block(self.model, expose=['mixture_weights', 'norm_factor', 'cnv_probs']),
                             init_loc_fn=self.init_fn())
        else:
            def guide_ret(*args, **kwargs):
                I, N = self._data['data'].shape
                batch = N if self._params['batch_size'] else self._params['batch_size']

                kappa = pyro.param('param_kappa', lambda: dist.Uniform(0, 1).sample([self._params['T'] - 1]), constraint=constraints.positive)
                cnv_mean = pyro.param("param_cnv_mean", lambda: self.create_gaussian_init_values(),
                                      constraint=constraints.positive)
                cnv_var = pyro.param("param_cnv_var", lambda: torch.ones(1) * self._params['cnv_var'],
                                     constraint=constraints.positive)
                gamma_scale = pyro.param("param_gamma_scale", lambda: torch.mean(
                    self._data['data'] / (2 * self._data['mu'].reshape(self._data['data'].shape[0], 1)), axis=0)  *  self._params['gamma_multiplier'],
                                         constraint=constraints.positive)
                gamma_rate = pyro.param("param_rate", lambda: torch.ones(1)  *  self._params['gamma_multiplier'],
                                        constraint=constraints.positive)
                param_weights = pyro.param("param_weights", lambda: torch.ones(self._params['T']) / self._params['T'],
                                           constraint=constraints.simplex)

                with pyro.plate("beta_plate", self._params['T'] - 1):
                    pyro.sample("mixture_weights", dist.Beta(1, kappa))

                with pyro.plate('segments', I):
                    with pyro.plate('components', self._params['T']):
                        pyro.sample('cnv_probs', dist.LogNormal(torch.log(cnv_mean), cnv_var))

                with pyro.plate("data2", N, batch):
                    pyro.sample('norm_factor', dist.Gamma(gamma_scale, gamma_rate).expand([N]))

            return guide_ret

    def create_gaussian_init_values(self):
        init = torch.zeros(self._params['T'], self._data['data'].shape[0])
        for i in range(len(self._data['pld'])):
            for k in range(self._params['T']):
                if k == 0:
                    init[k, i] = torch.ceil(self._data['pld'][i])
                else:
                    init[k, i] = torch.floor(self._data['pld'][i])
        return init

    def full_guide(self, MAP = False , *args, **kwargs):
        def full_guide_ret(*args, **kargs):
            I, N = self._data['data'].shape
            batch = N if self._params['batch_size'] else self._params['batch_size']

            with poutine.block(hide_types=["param"]):  # Keep our learned values of global parameters.
                self.guide(MAP)()
            with pyro.plate('data', N, batch):
                assignment_probs = pyro.param('assignment_probs', torch.ones(N, self._params['T']) / self._params['T'],
                                              constraint=constraints.unit_interval)
                pyro.sample('assignment', dist.Categorical(assignment_probs), infer={"enumerate": "parallel"})

        return full_guide_ret



    def init_fn(self):
        def init_function(site):
            if site["name"] == "cnv_probs":
                return self.create_gaussian_init_values()
            if site["name"] == "mixture_weights":
                return dist.Uniform(0, 1).sample([self._params['T'] - 1])
            if site["name"] == "norm_factor":
                return torch.mean(self._data['data'] / (2 * self._data['mu'].reshape(self._data['data'].shape[0],1)), axis=0)
            raise ValueError(site["name"])
        return init_function
