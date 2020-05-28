import pyro
import pyro.distributions as dist
import numpy as np
import torch
from models.Model import Model
from pyro.ops.indexing import Vindex
from pyro import poutine
from pyro.infer.autoguide import AutoDelta
from torch.distributions import constraints
import torch.nn.functional as F




# A simple mixture model for CNV inference, it assumes independence among the different segments, needs to be used after
# calling CNV regions with bulk DNA. CNVs modelled as LogNormal variables

# TODO: add support for joint inference with bulk counts (or allelic frequencies)


class MixtureGaussianDMP(Model):

    params = {'T' : 6, 'cnv_mean' : 2, 'cnv_var' :0.6, 'theta_scale' : 3, 'theta_rate' : 1, 'batch_size' : None,
            'mixture' : None}
    data_name = set(['data', 'mu','pld', 'segments'])


    def __init__(self, data_dict):
        self.params['mixture'] = 1 / torch.ones(self.params['T'])
        self._params = self.params
        super().__init__(data_dict, self.data_name)

    def mix_weights(self,beta):
        beta1m_cumprod = (1 - beta).cumprod(-1)
        return F.pad(beta, (0, 1), value=1) * F.pad(beta1m_cumprod, (1, 0), value=1)

    def model(self,*args, **kwargs):
        I, N = self._data['data'].shape
        T =  self._params['K']
        with pyro.plate("beta_plate", T - 1):
            beta = pyro.sample("mixture_weights", dist.Beta(1, 0.0001))

        with pyro.plate('segments', I):
            with pyro.plate('components', self._params['K']):
                cc = pyro.sample('cnv_probs', dist.LogNormal(np.log(self._params['pld']), self._params['cnv_var']))

        with pyro.plate('data', N, self._params['batch_size']):
            assignment = pyro.sample('assignment', dist.Categorical(self.mix_weights(beta)), infer={"enumerate": "parallel"})
            theta = pyro.sample('norm_factor', dist.Gamma(self._params['theta_scale'], self._params['theta_rate']))
            for i in pyro.plate('segments2', I):
                pyro.sample('obs_{}'.format(i), dist.Poisson((Vindex(cc)[assignment,i] * theta * self._data['mu'][i]
                                                              )
                                                             + 1e-8), obs=self.params['data'][i, :])

    def guide(self,*args, **kwargs):
        return AutoDelta(poutine.block(self.model, expose=['mixture_weights', 'norm_factor', 'cnv_probs']), init_loc_fn=self.init_fn())

    def create_gaussian_init_values(self):
        init = torch.zeros(self._params['T'], self._data['segments'])
        for i in range(len(self._data['pld'])):
            for k in range(self._params['T']):
                if k == 0:
                    init[k, i] = torch.ceil(self._data['pld'][i])
                else:
                    init[k, i] = torch.floor(self._data['pld'][i])
        return init

    def final_guide(self, MAPs):
        def full_guide(self, *args, **kargs):
            I, N = self.params['data'].shape
            with poutine.block(hide=["cnv_probs", "norm_factor", "mixture_weights"]):
                pyro.sample('mixture_weights',
                                      dist.Dirichlet(MAPs['mixture_weights'].detach()))
                with pyro.plate('segments', I):
                    with pyro.plate('components', self.params['K']):
                         pyro.sample("cnv_probs", dist.Delta(
                            MAPs['cnv_probs'].detach()))

                with pyro.plate('data', N, self.params['batch_size']):
                    assignment_probs = pyro.param('assignment_probs',
                                                  torch.ones(N, self.params['K']) / self.params['K']
                                                  , constraint=constraints.unit_interval)
                    pyro.sample('assignment', dist.Categorical(assignment_probs),
                                             infer={"enumerate": "parallel"})
                    pyro.sample('norm_factor',
                                dist.Delta(MAPs['norm_factor'].detach()))


        return full_guide



    def init_fn(self):
        def init_function(site):
            if site["name"] == "cnv_probs":
                return self.create_gaussian_init_values()
            if site["name"] == "mixture_weights":
                return dist.Beta(1, 0.8).mean.repeat((self.params['K']-1))
            if site["name"] == "norm_factor":
                return torch.mean(self.params['data'] / (2 * self.params['mu']), axis=0)
            raise ValueError(site["name"])
        return init_function

    def write_results(self, MAPs, prefix, trace=None, cell_ass = None):

        assert trace is not None or cell_ass is not None
        if cell_ass is not None:
            cell_assig = cell_ass
        else:
            cell_assig = trace.nodes["assignment"]["value"]

        np.savetxt(prefix + "cell_assignmnts.txt", cell_assig.numpy(), delimiter="\t")

        for i in MAPs:
            if i == "cnv_probs":
                np.savetxt(prefix + i + ".txt", MAPs[i].detach().numpy(), delimiter="\t")
            else:
                if i == "mixture_weights":
                    np.savetxt(prefix + i + ".txt", MAPs[i].detach().numpy(), delimiter="\t")
                else:
                    np.savetxt(prefix + i + ".txt", self.mix_weights(MAPs[i]).detach().numpy(), delimiter="\t")


        np.savetxt(prefix + "cnvs_inf.txt", torch.round(MAPs['cnv_probs']).detach().numpy(), delimiter="\t")
