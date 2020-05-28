from pyro.infer import SVI, infer_discrete, TraceEnum_ELBO, TraceGraph_ELBO
import pyro
from pyro import poutine
from pyro.optim import ClippedAdam


class Interface:
    """


    """
    def __init__(self,model = None, optimizer = None, loss = None, inf_type = SVI):
        self._model_fun = model
        self._optimizer = optimizer
        self._loss = loss
        self._model = None
        self._inf_type = SVI
        self._model_trained = None
        self._guide_trained = None
        self._loss_trained = None
        self._MAP = None


    def __repr__(self):

        if self._model is None:
            dictionary = {"Model": self._model_fun,
                    "Optimizer": self._optimizer,
                    "Loss": self._loss,
                    "Inference strategy": self._inf_type
                    }
        else:
            dictionary = {"Model" : self._model_fun,
                    "Data" : self._model._data,
                    "Model params": self._model._params,
                    "Optimizer" :self._optimizer,
                    "Loss" : self._loss,
                    "Inference strategy" :  self._inf_type
                    }

        return "\n".join("{}:\t{}".format(k, v) for k, v in dictionary.items())

    def initialize_model(self, data):
        assert self._model_fun is not None
        self._model = self._model_fun(data)

    def set_optimizer(self, optimizer):
        self._optimizer = optimizer

    def set_model(self, model):
        self._model_fun = model

    def set_loss(self, loss):
        self._loss = loss

    def set_model_params(self, param_dict):
        self._model.set_params(param_dict)

    def run(self, steps,param_optimizer = {'lr' : 0.05}, param_loss = None, seed = 3, MAP = False, verbose = True):

        pyro.set_rng_seed(seed)
        pyro.clear_param_store()
        model = self._model.model
        guide = self._model.guide(MAP)
        self._MAP = MAP
        optim = self._optimizer(param_optimizer)
        elbo = self._loss(param_loss) if param_loss is not None else self._loss()
        svi = self._inf_type(model, guide, optim, elbo)
        num_observations = self._model._data['data'].shape[1]
        loss = [None] * steps
        print('Running {} on {} cells for {} steps'.format(
            type(self._model).__name__, num_observations, steps))
        for step in range(steps):
            loss[step] = svi.step()

            if verbose:
                print('{: >5d}\t{}'.format(step, loss[step] / num_observations))
            else:
                if (step+1) % 10 == 0:
                    print('.' , end='')
                if (step+1) % 100 == 0:
                    print("\n")

        self._model_trained = model
        self._guide_trained = guide
        self._loss_trained = loss
        print("Done!")
        return loss


    def _get_params_no_autoguide(self):

        param_names = pyro.get_param_store().match("param")
        res = {nms: pyro.param(nms) for nms in param_names}
        return res

    def learned_parameters(self, posterior = False ,optim = ClippedAdam, loss = TraceEnum_ELBO,param_optimizer = {"lr" : 0.05}, param_loss = None, steps = 200, verbose = False):
        if self._MAP:
            params = self._guide_trained()

        else:
            params = self._get_params_no_autoguide()


        print("Computing assignment probabilities")
        if posterior:
            discrete_params = self.inference_categorical_posterior(optim, loss, param_optimizer, param_loss, steps, verbose)

        else:
            discrete_params = self.inference_categorical_MAP()

        trained_params_dict = {i : params[i].detach().numpy() for i in params}
        trained_params_dict['cell_assignmnts'] = discrete_params.numpy()

        return trained_params_dict



    def inference_categorical_MAP(self):

        guide_trace = poutine.trace(self._guide_trained).get_trace()  # record the globals
        trained_model = poutine.replay(self._model_trained, trace=guide_trace)  # replay the globals

        # Recover the enumerated categorical variables

        inferred_model = infer_discrete(trained_model, temperature=0, first_available_dim=-2)
        trace = poutine.trace(inferred_model).get_trace()

        return trace.nodes["assignment"]["value"]

    def inference_categorical_posterior(self, optim = ClippedAdam, loss = TraceEnum_ELBO,param_optimizer = {'lr' : 0.05}, param_loss = None, steps = 300, verbose = False):

        full_guide = self._model.full_guide(self._MAP)
        optim_discr = optim(param_optimizer)
        elbo_discr = loss(param_loss) if param_loss is not None else loss()
        num_observations = self._model._data['data'].shape[1]

        svi2 = SVI( self._model.model, full_guide, optim_discr, loss=elbo_discr)

        for i in range(steps):
            loss = svi2.step()
            if verbose:
                print('{: >5d}\t{}'.format(i, loss / num_observations))
            else:
                if (i+1) % 10 == 0:
                    print('.', end='')
                if (i+1) % 100 == 0:
                    print("\n")
        assignment_probs = pyro.param('assignment_probs').detach()
        return assignment_probs

    def save_results(self, MAPs, prefix = "out", *args, **kwargs):
        self._model.write_results(prefix,MAPs)








