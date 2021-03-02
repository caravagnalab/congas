""" Utils class

A set of utils function to run automatically an enetire inference cycle, plotting and saving results.

"""

import matplotlib.pyplot as plt
import pandas as pd
import os
from congas.Interface import Interface
import numpy as np
from pyro.optim import ClippedAdam
from pyro.infer import SVI, TraceEnum_ELBO
import torch

def plot_loss(loss, save = False, output = "run1"):
    plt.plot(loss)
    plt.title("ELBO")
    plt.xlabel("step")
    plt.ylabel("loss")
    if(save):
        plt.savefig(output + "_ELBO.png")


def dict_to_tensor(dict):
    for k,v in dict.items():
        if not torch.is_tensor(v):
            dict[k] = torch.tensor(v)

def run_analysis(data_dict,model , optim = ClippedAdam, elbo = TraceEnum_ELBO, inf_type = SVI,steps = 500, lr = 0.01, param_dict = {},MAP = True, seed = 3):

    """ Run an entire analysis with the minimum amount of parameters

    Simple function to run an entire step of inference and get the learned parameters back, less customizable than using
    directly the :class:`~congas.core.Interface` , but still should satisfy most of hte user.
    Look at the R interface for even a easier


    Args:
        data_dict: dictionary with parameters
        model: a model from one in congas.models
        optim: an optimizer from pyro.optim
        elbo: a loss function from pyro.infer
        inf_type: SVI or NUTS (Hemiltonian MCMC)
        steps: number of inference steps
        lr: learning rate
        param_dict: parameters for the model, look at the model documentation if you want to change them
        MAP: perform MAP over the last layer of random variable in the model or learn the parameters of the distribution
        seed: seed for pyro.set_rng_seed
        step_post: steps if learning also posterior probabilities

    Returns:
        dict: dictionary of parameters:value
        list: loss (divided by sample size)  for every time step (not the one for posteriors)

    """



    interface = Interface(model, optim, elbo, inf_type)

    dict_to_tensor(data_dict)
    dict_to_tensor(param_dict)
    interface.initialize_model(data_dict)
    interface.set_model_params(param_dict)

    loss = interface.run(steps= steps, seed=seed, param_optimizer={'lr' : lr}, MAP = MAP)
    parameters = interface.learned_parameters()

    return parameters, loss

def load_simulation_seg(dir, prefix):

    """ Read data from companion R package simulation

    A function to read the

    Args:
        dir: directory where the simulation files are stored
        prefix:

    Returns:

    """

    data = pd.read_csv(dir + os.sep + prefix + "_data.csv")
    cnv = pd.read_csv(dir + os.sep + prefix + "_cnv.csv")
    data = torch.tensor(data.values, dtype=torch.float32).t()
    segments, num_observations = data.shape
    ploidy = torch.tensor(cnv["ploidy_real"], dtype=torch.float32)
    mu = torch.tensor(cnv["mu"])

    return {"data" : data, "pld" : ploidy, "segments": segments,"mu" : mu}





def write_results(params, prefix, new_dir = False, dir_pref = None):
    """ Write parameters

    This function writes the parameters appending a prefix and optionally in a new directory

    Args:
        params: parameters dictionary
        prefix: prefix to append to the filenames
        new_dir: create a new directory or use an exsisting ones
        dir_pref: name of the directory

    """

    if (new_dir):
        try:
            os.mkdir(dir_pref)
        except FileExistsError:
            print("Directory already existing, saving there", flush=True)

        out_prefix = "." + os.sep + dir_pref + os.sep + prefix + "_"
    else:
        out_prefix = prefix + "_"

    for i in params:
            np.savetxt(out_prefix + i + ".txt", params[i], delimiter="\t")

def log_sum_exp(args):
    c = torch.amax(args, dim=0)
    return c + torch.log(torch.sum(torch.exp(args - c), axis=0))

def entropy(x):
    entr = torch.zeros([x.shape[0], x.shape[1]])
    for k in range(x.shape[0]):
        for i in range(x.shape[1]):
            for h in range(x.shape[2]):
                entr[k,i] += x[k,i,h] + torch.log(x[k,i,h])
    return entr.sum()


