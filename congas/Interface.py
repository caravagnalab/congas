""" Standardize models execution.

The core of the package just provides class with functions to make every model behave in the same way

"""



from pyro.infer import SVI, infer_discrete, TraceEnum_ELBO
import pyro
from pyro import poutine
from pyro.optim import ClippedAdam
import numpy as np
import torch.nn.functional as F
from tqdm import trange



class Interface:
    """ The interface class for all the congas models.

    Basically it takes a model, am optimizer and a loss function and provides a functions to run the inference and get the parameters
    back masking the differences between models.


    """
    def __init__(self,model = None, optimizer = None, loss = None, inf_type = SVI):
        self._model_fun = model
        self._optimizer = optimizer
        self._loss = loss
        self._model = None
        self._inf_type = inf_type
        self._model_trained = None
        self._guide_trained = None
        self._loss_trained = None
        self._model_string = None
        self._MAP = False
        self._Hmm = False


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
        self._model_string = type(self._model).__name__

    def set_optimizer(self, optimizer):
        self._optimizer = optimizer

    def set_model(self, model):
        self._model_fun = model

    def set_loss(self, loss):
        self._loss = loss

    def set_model_params(self, param_dict):
        self._model.set_params(param_dict)

    def run(self, steps,param_optimizer = {'lr' : 0.05}, param_loss = None, seed = 3, MAP = False):

        """ This function runs the inference of non-categorical parameters

          This function performs a complete inference cycle for the given tuple(model, optimizer, loss, inference modality).
          For more info about the parameters for the loss and the optimizer look
          at `Optimization <http://docs.pyro.ai/en/stable/optimization.html>`_.
          and `Loss <http://docs.pyro.ai/en/stable/inference_algos.html>`_.

          Not all the the combinations Optimize-parameters and Loss-parameters have been tested, so something may
          not work (please open an issue on the GitHub page).


          Args:
              steps (int): Number of steps
              param_optimizer (dict):  A dictionary of paramaters:value for the optimizer
              param_loss (dict): A dictionary of paramaters:value for the loss function
              seed (int): seed to be passed to  pyro.set_rng_seed
              MAP (bool): Perform learn a Delta distribution over the outer layer of the model graph
              verbose(bool): show loss for each step, if false the functions just prints a progress bar
              BAF(torch.tensor): if provided use BAF penalization in the loss

          Returns:
              list: loss (divided by sample size) for each step


          """

        pyro.set_rng_seed(seed)
        pyro.clear_param_store()
        model = self._model.model
        guide = self._model.guide(MAP)
        self._MAP = MAP
        # Hmm have special parameters
        if "hmm" in self._model_string.lower():
            self._Hmm = True

        optim = self._optimizer(param_optimizer)
        elbo = self._loss(**param_loss) if param_loss is not None else self._loss()
        svi = self._inf_type(model, guide, optim, elbo)

        num_observations = self._model._data['data'].shape[1]
        num_segments = self._model._data['data'].shape[0]

        loss = [None] * steps
        print('Running {} on {} cells wiht {} segments for {} steps'.format(
           self._model_string, num_observations, num_segments,steps), flush=True)
        t = trange(steps, desc='Bar desc', leave=True)

        for step in t:
            loss[step] = svi.step() / num_observations
            elb = loss[step]
            t.set_description('ELBO: {:.9f}  '.format(elb))
            t.refresh()

        print("", flush = True)
        self._model_trained = model
        self._guide_trained = guide
        self._loss_trained = loss
        print("Done!", flush=True)
        return loss





    def _get_params_no_autoguide(self):

        """ Return the parameters that are not enumerated when we do full inference

            Returns:
              dict: parameter:value dictionary
        """

        param_names = pyro.get_param_store().match("param")
        res = {nms: pyro.param(nms) for nms in param_names}
        return res

    def learned_parameters(self):

        """ Return all the estimated  parameter values

            Calls the right set of function for retrieving learned parameters according to the model type
            If posterior=True all the other parameters are just passed to :func:`~congas.core.Interface.inference_categorical_posterior`

            Args:

              posterior: learn posterior assignement (if false estimate MAP via Viterbi-like MAP inference)


            Returns:
              dict: parameter:value dictionary
        """


        if self._MAP:
            params = self._guide_trained()
            if "DMP" in self._model_string:
                params['betas'] = params['mixture_weights'].clone().detach()
                params['mixture_weights'] = self._mix_weights(params['mixture_weights'])

        else:
            params = self._get_params_no_autoguide()


        print("Computing assignment probabilities", flush=True)
        discrete_params = self._model.calculate_cluster_assignements(params)



        trained_params_dict = {i : params[i].detach().numpy() for i in params}

        all_params =  {**trained_params_dict,**discrete_params}

        return all_params


    def _mix_weights(self, beta):
        """ Get mixture wheights form beta samples

        When using a stick-breaking process, transform the beta samples in effective mixture weights

        Args:

            beta: beta  from the stick-breaking process

        Returns:
            mixture_weights:

        """

        beta1m_cumprod = (1 - beta).cumprod(-1)
        mixture_weights = F.pad(beta, (0, 1), value=1) * F.pad(beta1m_cumprod, (1, 0), value=1)
        return mixture_weights




