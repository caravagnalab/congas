""" ANNEAL: rnA cNvs iNferEnce And cLustering (with Pyro)

A set of Pyro models and an interface for simultaneous CNV clustering and inference


"""


from congas.Interface import Interface
from congas.utils import *
import os

location = os.path.dirname(os.path.realpath(__file__))
my_file = os.path.join(location, 'data')

simulation_data = load_simulation_seg(my_file,"example1")

