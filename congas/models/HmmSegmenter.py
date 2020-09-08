import pyro
import pyro.distributions as dist
import torch
from congas.models.Model import Model
from pyro.ops.indexing import Vindex
from pyro import poutine
from pyro.infer.autoguide import AutoDelta



class HmmSegmenter(Model):
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

    params = {'init_probs': torch.tensor([0.1, 0.3, 0.2, 0.2, 0.1, 0.1]), 'hidden_dim': 6,
                  't':  1e-8}

    data_name = set(['data', 'dist'])

    def __init__(self, data_dict):
        self._params = self.params.copy()
        self._data = None
        super().__init__(data_dict, self.data_name)
        self._data['segments'] = self._data['data'].shape[0]

    def model(self, *args, **kwargs):
        I = self._data['segments']




        pi = pyro.sample("pi", dist.Dirichlet(self._params['init_probs']))


        probs_z = pyro.sample("cnv_probs",
                              dist.Dirichlet((1- self._params['t']) * torch.eye(self._params['hidden_dim']) + (
                                      self._params['t'])).to_event(1))
        probs_y =  torch.tensor([[2., 64., 32., 21.5, 16., 43.],[64., 64., 64., 64., 64., 64.]])


        z = pyro.sample("z_0", dist.Categorical(pi),
                infer={"enumerate": "parallel"})

        pyro.sample("y_{}".format(0), dist.Beta(probs_y[0, z], probs_y[1, z]),
                    obs=self._data['data'][0, 0])

        for i in pyro.markov(range(1,I)):
            z = pyro.sample("z_{}".format(i), dist.Categorical(Vindex(probs_z)[z]),
                            infer={"enumerate": "parallel"})


            pyro.sample("y_{}".format(i), dist.Beta(probs_y[0,z], probs_y[1,z]),
                        obs= self._data['data'][i,0])







    def guide(self,MAP = False,*args, **kwargs):
        return AutoDelta(poutine.block(self.model, expose=['pi',  'cnv_probs']),
                             init_loc_fn=self.init_fn())

    def init_fn(self):
        def init_function(site):
            if site["name"] == "cnv_probs":
                return (self._params['t'] * torch.eye(self._params['hidden_dim']) + (1- self._params['t']))
            if site["name"] == "pi":
                return self._params['init_probs']
            raise ValueError(site["name"])
        return init_function



