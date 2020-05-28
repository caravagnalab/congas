from abc import ABC, abstractmethod

class Model(ABC):

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

    @abstractmethod
    def init_fn(self):
        pass

    @abstractmethod
    def write_results(self, prefix):
        pass

    def set_params(self, params_dict):

        self._params.update(params_dict)

