import pyro
import pyro.distributions as dist
import torch
import numpy as np
from congas.models.Model import Model
from pyro.ops.indexing import Vindex
from pyro import poutine
from pyro.infer.autoguide import AutoDelta
from torch.distributions import constraints
from congas.utils import entropy, log_sum_exp


"""

CNVs events are modelled as Categorical variables, what is shared by the clusters is the class probabilities for a given CNV satet
but different cells in the same clusters are allowed to have different CNVs profiles. It tries to get a mean 


Model parameters:
    T = max number of clusters (default = 6)
    probs = prior probs for the Dirichlet modelling group CNV probabilities (default=torch.tensor([0.1,0.1,0.2,0.3,0.2,0.1]))
    hidden_dim = hidden dimensions (should be len(probs))
    theta_scale = scale for the normalization factor variable (default = 3)
    theta_rate = rate for the normalization factor variable (default = 1)
    batch_size = batch size (default = None)
    mixture = prior for the mixture weights (default = 1/torch.ones(K))
    gamma_multiplier = multiplier Gamma(rate * gamma_multiplier, shape  * gamma_multiplier) when we also want to 
    infer the shape and rate parameter (i.e. when MAP = FALSE) (default = 4)





TODO: 
    test on meaningful datasets
"""

class MixtureDirichlet(Model):

    params = {'K': 2, 'probs': torch.tensor([0.00001,0.25,0.5,0.25,0.05,0.05]), 'hidden_dim': 5,'theta_scale' : None, 'theta_rate': None, 'batch_size': None,
              'mixture': None, "nb_size_init": None, "binom_prior_limits" : [0.01,10000] }
    data_name = set(['data', 'mu', 'pld', 'segments', 'norm_factor'])

    def __init__(self, data_dict):
        self._params = self.params.copy()
        self._data = None
        super().__init__(data_dict, self.data_name)

    def model(self,*args, **kwargs):
        I, N = self._data['data'].shape
        batch = N if self._params['batch_size'] else self._params['batch_size']
        weights = pyro.sample('mixture_weights', dist.Dirichlet((1 / self._params['K']) * torch.ones(self._params['K'])))
        with pyro.plate('segments', I):
            size = pyro.sample("NB_size", dist.Uniform(self._params['binom_prior_limits'][0],
                                                        self._params['binom_prior_limits'][1]))
            segment_factor = pyro.sample('segment_factor',
                                         dist.Gamma(self._params['theta_scale'], self._params['theta_rate']))
            with pyro.plate('components', self._params['K']):
                cnv_probs = pyro.sample("CNV_probabilities", dist.Dirichlet(self._params['probs']))

        with pyro.plate('data', N, batch):
            assignment = pyro.sample('assignment', dist.Categorical(weights), infer={"enumerate": "parallel"})
            cc = pyro.sample('copy_number', dist.Categorical(Vindex(cnv_probs)[assignment,:, :]),
                             infer={"enumerate": "parallel"})
            mean = (cc * segment_factor * self._data['norm_factor'].reshape([1,N])) + 1e-8
            pyro.factor('obs', dist.NegativeBinomial(probs = mean / (mean + size), total_count = size).log_prob(self._data['data'])
                        - entropy(cnv_probs))

    def guide(self,MAP = False,  *args, **kwargs):
        return AutoDelta(poutine.block(self.model, expose=['mixture_weights', 'segment_factor', 'CNV_probabilities', 'NB_size']), init_loc_fn=self.init_fn())



    def full_guide(self, MAP = False , *args, **kwargs):
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

        bins = self._params['hidden_dim'] * 2
        low_prob = 1 / bins
        high_prob = low_prob * (self._params['hidden_dim'] + 1)
        init = torch.zeros(self._params['K'], self._data['segments'], self._params['hidden_dim'])

        for i in range(len(self._data['pld'])):
            for j in range(self._params['hidden_dim']):
                for k in range(self._params['K']):
                    if k == 0:
                        init[k, i, j] = high_prob if j == torch.ceil(self._data['pld'][i]) else low_prob
                    else:
                        init[k, i, j] = high_prob if j == torch.floor(self._data['pld'][i]) else low_prob

        return init

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

    def likelihood_model(self, segment_fact_marg, weights, size, probs_cnv):
        lk = torch.zeros(self._params['hidden_dim'], self._params['K'], self._data['data'].shape[1],
                         self._data['data'].shape[0])
        if self._params['K'] == 1:
            # norm_f = torch.sum(segment_fact_marg)
            for i in range(self._data['data'].shape[0]):
                for h in range(self._params['hidden_dim']):
                    mean = (segment_fact_marg[i, h] * self._data['norm_factor'])  # / norm_f
                    lk[h, 0, :, i] = dist.NegativeBinomial(probs=mean / (mean + size[i]), total_count=size[i]).log_prob(
                        self._data['data'][i, :]) + torch.log(probs_cnv[0, i, h])
            return lk

        for k in range(self._params['K']):
            # norm_f = torch.sum(segment_fact_marg[k,:])
            for i in range(self._data['data'].shape[0]):
                for h in range(self._params['hidden_dim']):
                    mean = (segment_fact_marg[i, h] * self._data['norm_factor'])  # / norm_f
                    lk[h, k, :, i] = dist.NegativeBinomial(probs=mean / (mean + size[i]), total_count=size[i]).log_prob(
                        self._data['data'][i, :]) + torch.log(probs_cnv[k, i, h])

        return log_sum_exp(lk)

    def likelihood(self, inf_params):
        I, N = self._data['data'].shape
        cat_vector = torch.tensor(np.arange(self._params['hidden_dim']) + 1, dtype=torch.float)
        segment_fact = torch.matmul(inf_params["segment_factor"].reshape([I, 1]),
                                    cat_vector.reshape([1, self._params['hidden_dim']]))

        lk = self.likelihood_model(segment_fact, inf_params["mixture_weights"], inf_params["NB_size"],
                                   inf_params["CNV_probabilities"])
        return lk

    def calculate_cluster_assignements(self, inf_params):

        lk = self.likelihood(inf_params)
        lk = torch.sum(lk, dim=2) + torch.log(inf_params["mixture_weights"]).reshape([self._params['K'], 1])

        summed_lk = log_sum_exp(lk)
        ret = lk - summed_lk
        res = {"assignment_probs": torch.exp(ret).detach().numpy()}
        return res