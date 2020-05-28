import pyro
import pyro.distributions as dist
import numpy as np
import torch
from models.Model import Model
from pyro.ops.indexing import Vindex
from pyro import poutine
from pyro.infer.autoguide import AutoDelta
from torch.distributions import constraints




# A simple mixture model for CNV inference, it assumes independence among the different segments, needs to be used after
# calling CNV regions with bulk DNA. CNVs modelled as LogNormal variables

# TODO: add support for joint inference with bulk counts (or allelic frequencies)


class MixtureGaussianEXP(Model):

    params_name = set(['data', 'mu', 'K', 'cnv_mean', 'cnv_var', 'theta_scale', 'theta_rate', 'batch_size', 'mixture', 'pld', 'segments'])

    def __init__(self, param_dict):
        super().__init__(param_dict, self.params_name)

    def model(self,*args, **kwargs):
        I, N = self.params['data'].shape
        weights = pyro.sample('mixture_weights', dist.Dirichlet(torch.ones(self.params['K'])))
        with pyro.plate('segments', I):
            with pyro.plate('components', self.params['K']):
                cc = pyro.sample('cnv_probs', dist.LogNormal(np.log(self.params['pld']), self.params['cnv_var']))

        with pyro.plate('data', N, self.params['batch_size']):
            assignment = pyro.sample('assignment', dist.Categorical(weights), infer={"enumerate": "parallel"})
            theta = pyro.sample('norm_factor', dist.Gamma(self.params['theta_scale'], self.params['theta_rate']))
            for i in pyro.plate('segments2', I):
                pyro.sample('obs_{}'.format(i), dist.Poisson((Vindex(cc)[assignment,i] * theta * self.params['mu'][i])
                                                             + 1e-8), obs=self.params['data'][i, :])

    def guide(self,*args, **kwargs):
        return AutoDelta(poutine.block(self.model, expose=['mixture_weights', 'norm_factor', 'cnv_probs']), init_loc_fn=self.init_fn())

    def create_gaussian_init_values(self):
        init = torch.zeros(self.params['K'], self.params['segments'])
        for i in range(len(self.params['pld'])):
            for k in range(self.params['K']):
                if k == 0:
                    init[k, i] = torch.ceil(self.params['pld'][i])
                else:
                    init[k, i] = torch.floor(self.params['pld'][i])
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
                return self.params['mixture']
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
                np.savetxt(prefix + i + ".txt", MAPs[i].detach().numpy(), delimiter="\t")

        np.savetxt(prefix + "cnvs_inf.txt", torch.round(MAPs['cnv_probs']).detach().numpy(), delimiter="\t")
