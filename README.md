# RNA CNVs Inference and Clustering


[![Build Status](https://travis-ci.org/Militeee/anneal.svg?branch=master)](https://travis-ci.org/Militeee/anneal)
[![codecov](https://codecov.io/gh/Militeee/anneal/branch/master/graph/badge.svg)](https://codecov.io/gh/Militeee/anneal)


A set of Pyro models and functions to infer CNA from scRNA-seq data. 
It comes with a companion R package (**in progress**) that works as an interface and provides preprocessing, simulation and visualization routines.


Currently providing:

- A mixture model on segments where CNV are modelled as LogNormal random variable (MixtureGaussian) 
- Same as above but the number of cluster is learned (MixtureGaussianDMP)
- A model where CNVs are modelled as outcome from Categorical distributions, clusters share the same parameters (MixtureDirichlet)
- A simple Hmm where CNVs are again categorical, but there is no clustering (SimpleHmm)
- The version of MixtureDirichlet but with temporal dependency  (HmmMixtureRNA)

Coming soon:
- A linear model in the emission that can account for known covariates
- The equivalent of MixtureGaussian but with CNVs as Categorical random variable
- A model on genes (all the other models assume a division in segments)

To install:

`$ pip install anneal`

To run a simple analysis on the example data

```
import anneal as an
data_dict = an.load_simulation_seg("data", "example1")
params, loss = an.run_analysis(data_dict,MixtureGaussian, steps=200, lr=0.05)
```


[Full Documentation](https://annealpyro.readthedocs.io/en/latest/)
