from abc import ABC, abstractmethod
import torch

class Model(ABC):
    """

    All the models in the package have more or less the same structure. Cells are assumed to come from different
    population based on their copy-number profiles. Each model treats the CNV in a different way, but the common idea is
    to have an explicit formula for the counts of a gene or a genomic segment and treat them as if they depends only on
    the CNV and a cell specific factor (for segments, for genes we also have a gene dependent effect).

    Moreover gene/segment counts are independent over the genome (or at least they hava a temporal dependency of first
    order). Given that we can write a mixture model factorizing the CNV for every segment.

    """

    def __init__(self, data_dict, data_name):
        data = {k: v for k, v in data_dict.items() if k in data_name}
        self._data = data
        super().__init__()

    @abstractmethod
    def model(self):
        pass

    @abstractmethod
    def guide(self):
        pass

    def set_params(self, params_dict):

        self._params.update(params_dict)
        if 'mixture' in self._params and self._params['mixture'] is None:
            self._params['mixture'] = 1 / (torch.ones(self._params['K']) * self._params['K'])

