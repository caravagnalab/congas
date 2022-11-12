from congas.utils import *
import numpy as np

def all_stopping_criteria(old, new, e, step):
    old = collect_params(old)
    new = collect_params(new)
    diff_mix = np.abs(old - new) / np.abs(old)
    
    if np.all(diff_mix < e):
        return [True, diff_mix]
    return [False, diff_mix]