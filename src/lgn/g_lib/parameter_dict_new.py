import torch
from torch.nn import Parameter, ParameterDict

class ParameterDictNew(ParameterDict):
    """ We modify ParameterDict so that keys can be non-strings.
    In our case, tuples of integers, but this code is universal."""

    def __getitem__(self, key):
            return self._parameters[str(key)]

    def __setitem__(self, key, parameter):
            self.register_parameter(str(key), parameter)

    def keys(self):
        return set(map(eval,self._parameters.keys()))

    def items(self):
        return set(map(lambda k,v: (eval(k),v), self._parameters.items()))
