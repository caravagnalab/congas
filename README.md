# Copy number genotyping from scRNA sequencing


[![Build Status](https://travis-ci.org/Militeee/anneal.svg?branch=master)](https://travis-ci.org/Militeee/congas)
[![codecov](https://codecov.io/gh/Militeee/anneal/branch/master/graph/badge.svg)](https://codecov.io/gh/Militeee/congas)


A set of Pyro models and functions to infer CNA from scRNA-seq data. 
It comes with a companion [R package](https://github.com/caravagnalab/rcongas) that works as an interface and provides preprocessing, simulation and visualization routines.
We suggest to use the R package directly as this serves mosttly as a backend for computations.


Currently providing:

- A mixture model on segments where CNV are modelled as LogNormal random variable (MixtureGaussian) 
- A mixture model on segments where CNV are modelled as Categorical random variable (MixtureCategorical) 
- A simple Hmm where CNVs are again categorical, but there is no clustering (SimpleHmm)

To install:

`$ pip install congas`

To run a simple analysis on the example data

```python
import congas as cn
from congas.models import MixtureGaussian
data_dict = cn.simulation_data
params, loss = cn.run_analysis(data_dict,MixtureGaussian, steps=200, lr=0.05)
```
