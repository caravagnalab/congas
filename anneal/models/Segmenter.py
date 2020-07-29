import pyro
import pyro.distributions as dist
import torch
from anneal.models.Model import Model
from pyro.ops.indexing import Vindex
from pyro import poutine
from pyro.infer.autoguide import AutoDelta



class Segmenter(Model):
    """

    An Hmm to segment normalized (against a wild-type reference) scRNA-seq count matrices


    Model parameters:
        T = max number of clusters (default = 6)
        init_probs = prior probs for initial state CNV probabilities (default=torch.tensor([0.1,0.1,0.2,0.3,0.2,0.1]))
        hidden_dim = hidden dimensions (should be len(probs))
        theta_scale = scale for the normalization factor variable (default = 3)
        theta_rate = rate for the normalization factor variable (default = 1)
        batch_size = batch size (default = None)
        t = probability of remaining in the same state (default=0.1)




    """

    params = {'init_probs': torch.tensor([0.1, 0.1, 0.5, 0.1, 0.1, 0.1]), 'hidden_dim': 6,
                  'theta_scale': 9, 'theta_rate': 3, 'batch_size': None ,
                  't':  0.999}
    data_name = set(['data', 'major', 'minor', 'dist'])

    def __init__(self, data_dict):
        self._params = self.params.copy()
        self._data = None
        super().__init__(data_dict, self.data_name)

    def model(self, *args, **kwargs):
        I, N = self._data['data'].shape



        batch = N if self._params['batch_size'] else self._params['batch_size']

        cnv_six_baf = pyro.param("baf6",   torch.ones(1))
         emission_baf1 = torch.Tensor([])
        emission_baf2 = torch.Tensor([])

        pi = pyro.sample("pi", dist.Dirichlet(self._params['init_probs']))

        z = pyro.sample("init_state", dist.Categorical(pi),
                        infer={"enumerate": "parallel"})
        with pyro.plate("data2", N, batch):
            theta = pyro.sample('norm_factor', dist.Gamma(self._params['theta_scale'], self._params['theta_rate']))

        for i in pyro.markov(range(I)):
            probs_z = pyro.sample("cnv_probs_{}".format(i),
                                  dist.Dirichlet(self._params['t'] * torch.eye(self._params['hidden_dim']) + (
                                              1 - self._params['t']))
                                  .to_event(1))
            z = pyro.sample("z_{}".format(i), dist.Categorical(Vindex(probs_z)[z]),
                            infer={"enumerate": "parallel"})
            with pyro.plate('data_{}'.format(i), N, batch):
                pyro.sample('obs_{}'.format(i), dist.Poisson((z * theta )
                                                                 + 1e-8), obs=self._data['data'][i,:])
            obs_baf =  self._data['minor'][i] / (self._data['major'][i] + self._data['minor'][i])
            if obs_baf > 0.05:





    def guide(self,MAP = False,*args, **kwargs):
        return AutoDelta(poutine.block(self.model, expose=['pi', 'norm_factor', 'cnv_probs']),
                             init_loc_fn=self.init_fn())

    def init_fn(self):
        def init_function(site):
            I, N = self._data['data'].shape
            if site["name"] == "cnv_probs":
                return (self._params['t'] * torch.eye(self._params['hidden_dim']) + (1- self._params['t']))
            if site["name"] == "norm_factor":
                return torch.mean(self._data['data'] / (2 * self._data['mu'].reshape(self._data['data'].shape[0],1)), axis=0)
            if site["name"] == "pi":
                return self._params['init_probs']
            raise ValueError(site["name"])
        return init_function



