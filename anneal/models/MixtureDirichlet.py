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
# calling CNV regions with bulk DNA.

# TODO: add support for joint inference with bulk counts (or allelic frequencies)


class MixtureDirichlet(Model):

    params_name = set(['data', 'mu', 'K', 'hidden_dim', 'probs', 'theta_scale', 'theta_rate', 'batch_size', 'mixture','pld', 'segments', 'num_observations', 'mask', 'segments_or'])

    def __init__(self, param_dict):
        if param_dict['probs'] is None:
            param_dict['probs'] = torch.ones(param_dict['hidden_dim']) * 1/param_dict['hidden_dim']
        super().__init__(param_dict, self.params_name)

    def model(self,*args, **kwargs):
        I, N = self.params['data'].shape
        weights = pyro.sample('mixture_weights', dist.Dirichlet((1 / self.params['K']) * torch.ones(self.params['K'])))
        with pyro.plate('segments', I):
            with pyro.plate('components', self.params['K']):
                cnv_probs = pyro.sample("cnv_probs", dist.Dirichlet(self.params['probs'] * 1/torch.ones(self.params['hidden_dim'])))
        with pyro.plate('data', N, self.params['batch_size']):
            assignment = pyro.sample('assignment', dist.Categorical(weights), infer={"enumerate": "parallel"})
            theta = pyro.sample('norm_factor', dist.Gamma(self.params['theta_scale'], self.params['theta_rate']))
            for i in pyro.plate('segments2', I):
                cc = pyro.sample('copy_number_{}'.format(i), dist.Categorical(Vindex(cnv_probs)[assignment,i, :]),
                                 infer={"enumerate": "parallel"})
                pyro.sample('obs_{}'.format(i), dist.Poisson((cc * theta * self.params['mu'][i]) + 1e-8), obs=self.params['data'][i, :])

    def guide(self, *args, **kwargs):
        return AutoDelta(poutine.block(self.model, expose=['mixture_weights', 'norm_factor', 'cnv_probs']), init_loc_fn=self.init_fn())


    def final_guide(self, MAPs):
        def full_guide(self, *args, **kargs):
            I, N = self.params['data'].shape
            with poutine.block(hide=["cnv_probs","norm_factor", "mixture_weights"]):
                weights = pyro.sample('mixture_weights',
                                      dist.Dirichlet(MAPs['mixture_weights'].detach()))
                with pyro.plate('segments', I):
                    with pyro.plate('components', self.params['K']):
                        cnv_probs = pyro.sample("cnv_probs", dist.Dirichlet(
                            MAPs['cnv_probs'].detach()))
                with pyro.plate('data', N, self.params['batch_size']):
                    assignment_probs = pyro.param('assignment_probs',
                                                  torch.ones(N, self.params['K']) / self.params['K']
                                                  , constraint=constraints.unit_interval)
                    assignment = pyro.sample('assignment', dist.Categorical(assignment_probs), infer={"enumerate": "parallel"})
                    pyro.sample('norm_factor',
                                        dist.Delta(MAPs['norm_factor'].detach()))
                    for i in pyro.plate('segments2', I):
                        pyro.sample('copy_number_{}'.format(i),
                                         dist.Categorical(Vindex(cnv_probs)[assignment, i, :]),
                                         infer={"enumerate": "parallel"})

        return full_guide


    def create_dirichlet_init_values(self):

        bins = self.params['hidden_dim'] * 2
        low_prob = 1 / bins
        high_prob = low_prob * (self.params['hidden_dim'] + 1)
        init = torch.zeros(self.params['K'], self.params['segments'], self.params['hidden_dim'])

        for i in range(len(self.params['pld'])):
            for j in range(self.params['hidden_dim']):
                for k in range(self.params['K']):
                    if k == 0:
                        init[k, i, j] = high_prob if j == torch.ceil(self.params['pld'][i]) else low_prob
                    else:
                        init[k, i, j] = high_prob if j == torch.floor(self.params['pld'][i]) else low_prob

        return init

    def init_fn(self):
        def init_function(site):
            if site["name"] == "cnv_probs":
                return self.create_dirichlet_init_values()
            if site["name"] == "mixture_weights":
                return self.params['mixture']
            if site["name"] == "norm_factor":
                return torch.mean(self.params['data'] / (2 * self.params['mu']), axis=0)
            raise ValueError(site["name"])
        return init_function

    def write_results(self, MAPs, prefix, trace=None, cell_ass = None):

        assert trace is not None or cell_ass is not None
        if cell_ass is not None :
            cell_assig = cell_ass
        else:
            cell_assig = trace.nodes["assignment"]["value"]

        cnvs_table = torch.zeros((self.params['num_observations'], self.params['segments_or']))

        for i in range(self.params['segments_or']):
            if i in self.params['mask']:
                cnvs_table[:, i] = trace.nodes['copy_number_{}'.format((self.params['mask'] == i).nonzero().item())]['value']
            else:
                cnvs_table[:, i] = torch.ones(1000) * self.params['mu_or'][i]

        np.savetxt(prefix + "cell_assignmnts.txt", cell_assig.numpy(), delimiter="\t")
        np.savetxt(prefix + "cnvs_table.txt", cnvs_table.numpy(), delimiter="\t")
        np.savetxt(prefix + "cnvs_inf.txt", torch.argmax(MAPs["cnv_probs"], dim=2).numpy(), delimiter="\t")

        for i in MAPs:
            if i == "cnv_probs":
                for k in range(MAPs[i].shape[0]):
                    np.savetxt(prefix + i + "_" + str(k) + ".txt", MAPs[i][k].detach().numpy(), delimiter="\t")
            else:
                np.savetxt(prefix + i + ".txt", MAPs[i].detach().numpy(), delimiter="\t")

