import pyro
import pyro.distributions as dist
import torch
from congas.models.Model import Model
from pyro.ops.indexing import Vindex
from pyro import poutine
from pyro.infer.autoguide import AutoDelta
from torch.distributions import constraints
from sklearn.cluster import KMeans





class MixtureGaussianEXP(Model):

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

    params = {'K': 2, 'cnv_sd': 0.6, 'theta_scale': 1, 'theta_rate': 1, 'batch_size': None,
              'mixture':  None, 'norm_init_factors': None, 'kmeans' : True, 'seg_sd' : 0.1}
    data_name = set(['data', 'pld'])

    def __init__(self, data_dict):

        self._params = self.params.copy()
        self._data = None
        super().__init__(data_dict, self.data_name)

    def model(self,*args, **kwargs):
        I, N = self._data['data'].shape
        batch = N if self._params['batch_size'] else self._params['batch_size']

        weights = pyro.sample('mixture_weights', dist.Dirichlet(torch.ones(self._params['K'])))

        seg_mean = torch.mean(self._data['data'] / self._data['pld'].reshape([I,1]) , axis = 1)

        with pyro.plate('segments', I):
            segment_mean = pyro.sample('segment_mean', dist.LogNormal(torch.log(seg_mean) - self._params['seg_sd']**2 / 2, self._params['seg_sd']))
            with pyro.plate('components', self._params['K']):
                cc = pyro.sample('cnv_probs', dist.LogNormal(torch.log(self._data['pld']) - self._params['cnv_sd']**2 / 2, self._params['cnv_sd']))

        with pyro.plate("data2", N, batch):
            theta = pyro.sample('norm_factor', dist.Gamma(self._params['theta_scale'], self._params['theta_rate']))

        with pyro.plate('data', N, batch):

            assignment = pyro.sample('assignement', dist.Categorical(weights), infer={"enumerate": "parallel"})

            for i in pyro.plate('segments2', I):
                pyro.sample('obs_{}'.format(i), dist.Poisson(((Vindex(cc)[assignment,i] * theta * segment_mean[i])
                                                             + 1e-8)), obs=self._data['data'][i, :])


    def guide(self,MAP = False,*args, **kwargs):
        if(MAP):
            return AutoDelta(poutine.block(self.model, expose=['mixture_weights', 'norm_factor', 'cnv_probs', 'segment_mean']),
                             init_loc_fn=self.init_fn())
        else:
            def guide_ret(*args, **kwargs):
                I, N = self._data['data'].shape

                seg_mean = torch.mean(self._data['data'] / self._data['pld'].reshape([I, 1]), axis=1)

                batch = N if self._params['batch_size'] else self._params['batch_size']

                param_weights = pyro.param("param_mixture_weights", lambda: torch.ones(self._params['K']) / self._params['K'],
                                           constraint=constraints.simplex)
                cnv_mean = pyro.param("param_cnv_probs", lambda: self.create_gaussian_init_values(),
                                         constraint=constraints.positive)
                cnv_var = pyro.param("param_cnv_var", lambda: torch.ones(I) * self._params['cnv_sd'],
                                      constraint=constraints.positive)

                seg_var = pyro.param("param_seg_var", lambda: torch.ones(I) * self._params['cnv_sd'],
                                     constraint=constraints.positive)
                seg_mean = pyro.param("param_seg_mean", lambda: seg_mean)

                gamma_scale = pyro.param("param_norm_factor", lambda: torch.sum(self._data['data'], axis = 0) / torch.sum(seg_mean),
                                   constraint=constraints.positive)

                pyro.sample('mixture_weights', dist.Dirichlet(param_weights))

                with pyro.plate('segments', I):

                    pyro.sample('segment_mean',  dist.LogNormal(torch.log(seg_mean) - seg_var ** 2 / 2, seg_var))

                    with pyro.plate('components', self._params['K']):
                        pyro.sample('cnv_probs', dist.LogNormal(torch.log(cnv_mean) - cnv_var ** 2 / 2, cnv_var))

                with pyro.plate("data2", N, batch):
                    pyro.sample('norm_factor', dist.Delta(gamma_scale))


            return guide_ret


    def create_gaussian_init_values(self):
        init = torch.zeros(self._params['K'], self._data['data'].shape[0])
        seg_mean = torch.mean(self._data['data'] / self._data['pld'].reshape([self._data['data'].shape[0], 1]), axis=1)
        for i in range(len(self._data['pld'])):
            if self._params['kmeans']:
                if self._params['norm_init_factors'] is None:
                    norm = torch.sum(self._data['data'], axis = 0) / torch.sum(seg_mean)
                    X = (self._data['data'][i,:].detach().numpy() / (norm * seg_mean[i]))
                else:
                    X = (self._data['data'][i,:] / self._params['norm_init_factors']) / seg_mean[i]
                X = X.detach().numpy()
                km = KMeans(n_clusters=self._params['K'], random_state=0).fit(X.reshape(-1,1))
                centers = torch.tensor(km.cluster_centers_).flatten()
                for k in range(self._params['K']):
                    init[k, i] = centers[k]
            else:
                for k in range(self._params['K']):
                    init[k, i] = dist.LogNormal(torch.log(self._data['pld']) - self._params['cnv_sd']**2 / 2, self._params['cnv_sd']).sample([self._params['K']])



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
                    seg_mean = torch.mean(self._data['data'] / self._data['pld'].reshape([self._data['data'].shape[0], 1]), axis=1)
                    return torch.sum(self._data['data'], axis = 0) / torch.sum(seg_mean)
                else:
                    return self._params['norm_init_factors']
            if site['name'] == "segment_mean":
                return torch.mean(self._data['data'] / self._data['pld'].reshape([self._data['data'].shape[0], 1]), axis=1)
            raise ValueError(site["name"])
        return init_function



