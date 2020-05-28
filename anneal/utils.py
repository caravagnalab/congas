import matplotlib.pyplot as plt
import pandas as pd
import torch
import os
import numpy as np
from anneal.core import *

def plot_loss(loss, save = False, output = ""):
    plt.plot(loss)
    plt.title("ELBO")
    plt.xlabel("step")
    plt.ylabel("loss")
    if(save):
        plt.savefig(output + "_ELBO.png")

def run_analysis(model, optim, elbo, inf_type, data_dict, param_dict = {}, posteriors = False, seed = 3, steps = 500, lr = 0.01):

    interface = Interface(model, optim, elbo, inf_type)
    interface.initialize_model(data_dict)
    interface.set_model_params(param_dict)
    loss = interface.run(steps, seed, {'lr' : lr})
    plot_loss(loss)

    interface.save_results(post)

def load_simulation_seg(dir, prefix):

    data = pd.read_csv(dir + "/" + prefix + "_data.csv")
    cnv = pd.read_csv(dir + "/" + prefix + "_cnv.csv")
    data = torch.tensor(data.values, dtype=torch.float32).t()
    segments, num_observations = data.shape
    ploidy = torch.tensor(cnv["ploidy_real"], dtype=torch.float32)
    mu = torch.tensor(cnv["mu"])
    # mu = mu.repeat((num_observations, 1)).t()

    return {"data" : data, "pld" : ploidy, "segments": segments,"mu" : mu}


def load_real_data_seg():
    pass


def write_results(params, prefix, new_dir = False, dir_pref = None):

    if (new_dir):
        try:
            os.mkdir(dir_pref)
        except FileExistsError:
            print("Directory already existing, saving there")

        out_prefix = "./" + dir_pref + "/" + prefix + "_"
    else:
        out_prefix = prefix + "_"

    for i in params:
            np.savetxt(out_prefix + i + ".txt", params[i], delimiter="\t")