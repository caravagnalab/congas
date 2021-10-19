import pyro
import pyro.distributions as dist
import torch
from congas.models.Model import Model
from pyro.ops.indexing import Vindex
from pyro import poutine
from pyro.infer.autoguide import AutoDelta
from torch.distributions import constraints
from sklearn.cluster import KMeans





class MixtureGaussianOld(Model):

    """

    A simple mixture model for CNV inference, it assumes independence among the different segments, needs to be used after
    calling CNV regions with bulk DNA or RNA. CNVs events are modelled as LogNormal distributions.


    Model parameters:
        - **K = number** of clusters (default = 2)
        - **cnv_var = var of the LogNorm prior (default = 0.6)
        - **theta_scale =** scale for the normalization factor variable (default = 3)
        - **theta_rate =** rate for the normalization factor variable (default = 1)
        - **batch_size =** batch size (default = None)
        - **mixture =** prior for the mixture weights (default = 1/torch.ones(K))





    TODO:
        add support for joint inference with bulk counts (or allelic frequencies)


    """

    params = {'K': 2, 'cnv_sd': 0.1, 'theta_scale': 3, 'theta_rate': 1, 'batch_size': None,
              'mixture':  None, 'norm_init_factors': None, 'kmeans' : True, 'norm_factor' : None, 'assignments' : None, 'cnv_locs' : None}
    data_name = set(['data', 'mu', 'pld'])

    def __init__(self, data_dict):

        self._params = self.params.copy()
        self._data = None
        super().__init__(data_dict, self.data_name)

    def model(self,*args, **kwargs):
        I, N = self._data['data'].shape
        batch = N if self._params['batch_size'] else self._params['batch_size']
        if self._params['assignments'] is None:
            weights = pyro.sample('mixture_weights', dist.Dirichlet(torch.ones(self._params['K'])))

        with pyro.plate('segments', I):
            with pyro.plate('components', self._params['K']):
                cc = pyro.sample('cnv_probs', dist.LogNormal(torch.log(self._data['pld']), self._params['cnv_sd']))

        with pyro.plate("data2", N, batch):
            if self._params['norm_factor'] is None:
                theta = pyro.sample('norm_factor', dist.Gamma(self._params['theta_scale'], self._params['theta_rate']))
            else:
                theta = torch.tensor(self._params['norm_factor'])

        with pyro.plate('data', N, batch):
            if self._params['assignments'] is None:
                assignment = pyro.sample('assignment', dist.Categorical(weights), infer={"enumerate": "parallel"})
            else:
                assignment = torch.tensor(self._params['assignments'])

            for i in pyro.plate('segments2', I):
                pyro.sample('obs_{}'.format(i), dist.Poisson(((Vindex(cc)[assignment,i] * theta * self._data['mu'][i])
                                                             + 1e-8)), obs=self._data['data'][i, :])

    def guide(self,MAP = False,*args, **kwargs):
        if(MAP):
            exposing = ['cnv_probs']

            if self._params['assignments'] is None:
                exposing.append('mixture_weights')
            if self._params['norm_factor'] is None:
                exposing.append('norm_factor')
            return AutoDelta(poutine.block(self.model, expose=exposing),
                             init_loc_fn=self.init_fn())
        else:
            def guide_ret(*args, **kwargs):
                I, N = self._data['data'].shape
                batch = N if self._params['batch_size'] else self._params['batch_size']

                if self._params['assignments'] is None:
                    param_weights = pyro.param("param_mixture_weights", lambda: torch.ones(self._params['K']) / self._params['K'],
                                           constraint=constraints.simplex)
                if self._params['cnv_locs'] is None:
                    cnv_mean = pyro.param("param_cnv_probs", lambda: self.create_gaussian_init_values(),
                                         constraint=constraints.positive)
                else:
                    cnv_mean = self._params['cnv_locs']
                cnv_var = pyro.param("param_cnv_var", lambda: torch.ones([self._params['K'], I]) * self._params['cnv_sd'],
                                      constraint=constraints.positive)
                if self._params['norm_factor'] is None:
                    gamma_scale = pyro.param("param_norm_factor", lambda: torch.mean(self._data['data'] /
                                                                                 (self._data['mu'].reshape(self._data['data'].shape[0],1)), axis=0),
                                   constraint=constraints.positive)

                if self._params['assignments'] is None:
                    pyro.sample('mixture_weights', dist.Dirichlet(param_weights))

                with pyro.plate('segments', I):
                    with pyro.plate('components', self._params['K']):
                        pyro.sample('cnv_probs', dist.LogNormal(torch.log(cnv_mean), cnv_var))

                with pyro.plate("data2", N, batch):
                    if self._params['norm_factor'] is None:
                        pyro.sample('norm_factor', dist.Delta(gamma_scale))



            return guide_ret


    def create_gaussian_init_values(self):
        init = torch.zeros(self._params['K'], self._data['data'].shape[0])
        for i in range(len(self._data['pld'])):
            if self._params['kmeans']:
                if self._params['norm_init_factors'] is None:
                    norm = torch.mean(self._data['data'] / (self._data['pld'].reshape(self._data['data'].shape[0],1) * self._data['mu'].reshape(self._data['data'].shape[0],1)), axis=0)
                    X = (self._data['data'][i,:].detach().numpy() / norm.detach().numpy()) / self._data["mu"][i]
                else:
                    X = (self._data['data'][i,:] / self._params['norm_init_factors']) / self._data["mu"][i]
                X = X.detach().numpy()
                km = KMeans(n_clusters=self._params['K'], random_state=0).fit(X.reshape(-1,1))
                centers = torch.tensor(km.cluster_centers_).flatten()
                for k in range(self._params['K']):
                    init[k, i] = centers[k]
            else:
                for k in range(self._params['K']):
                    init[k, i] = dist.LogNormal(torch.log(self._data['pld']), self._params['cnv_sd']).sample([self._params['K']])



        return init

    def full_guide(self, MAP = False , *args, **kwargs):
        def full_guide_ret(*args, **kargs):
            I, N = self._data['data'].shape
            batch = N if self._params['batch_size'] else self._params['batch_size']

            with poutine.block(hide_types=["param"]):  # Keep our learned values of global parameters.
                self.guide(MAP)()
            with pyro.plate('data', N, batch):
                assignment_probs = pyro.param('assignment_probs', torch.ones(N, self._params['K']) / self._params['K'],
                                              constraint=constraints.simplex)
                pyro.sample('assignment', dist.Categorical(assignment_probs), infer={"enumerate": "parallel"})

        return full_guide_ret



    def init_fn(self):
        def init_function(site):
            if site["name"] == "cnv_probs":
                return self.create_gaussian_init_values()
            if site["name"] == "mixture_weights":
                return self._params['mixture']
            if site["name"] == "norm_factor":
                if self._params['norm_init_factors'] is None:
                    return torch.mean(self._data['data'] /
                                      (self._data['pld'].reshape(self._data['data'].shape[0],1) * self._data['mu'].reshape(self._data['data'].shape[0],1)), axis=0)
                else:
                    return self._params['norm_init_factors']
            raise ValueError(site["name"])
        return init_function



