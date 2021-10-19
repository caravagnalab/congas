Copy number genotyping from scRNA sequencing
============================================

A set of Pyro models and functions to infer CNA from scRNA-seq data. It
comes with a companion R package (**in progress**) that works as an
interface and provides preprocessing, simulation and visualization
routines.

Currently providing:

-  A mixture model on segments where CNV are modelled as LogNormal
   random variable (MixtureGaussian)
-  Same as above but the number of cluster is learned
   (MixtureGaussianDMP)
-  A model where CNVs are modelled as outcome from Categorical
   distributions, clusters share the same parameters (MixtureDirichlet)
-  A simple Hmm where CNVs are again categorical, but there is no
   clustering (SimpleHmm)
-  The version of MixtureDirichlet but with temporal dependency
   (HmmMixtureRNA)

Coming soon: - NUTS support

To install:

``$ pip install congas``

To run a simple analysis on the example data

::

    import congas as cn
    from congas.models import MixtureGaussian
    data_dict = cn.simulation_data
    params, loss = cn.run_analysis(data_dict,MixtureGaussian, steps=200, lr=0.05)
