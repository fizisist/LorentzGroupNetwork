import torch
from torch.nn import Parameter
from lgn.g_lib import g_tensor, g_tau, parameter_dict_new

GTau = g_tau.GTau
GTensor = g_tensor.GTensor
ParameterDictNew = parameter_dict_new.ParameterDictNew

class GWeight(GTensor):
    """
    Core class for creating and tracking G Weights that
    are used to part-wise mix a :obj:`GVec`.

    At the core of each :obj:`GWeight` is a list of :obj:`torch.Tensors` with
    shape `(C_{out}, C_{in}, 2)`, where:

    * `C_{in}` is the channels/multiplicity (tau) of the input :obj:`GVec`.
    * `C_{out}` is the channels/multiplicity (tau) of the output :obj:`GVec`.
    * `2` corresponds to the real/imaginary parts of the complex dimension.

    Parameters
    ----------

    data : List of of `torch.Tensor` with appropriate shape
        Input of a G Weight object.
    """

    @property
    def zdim(self):
        return 0

    @property
    def bdim(self):
        return None

    @property
    def cdim(self):
        return None

    @property
    def rdim(self):
        return None

    @staticmethod
    def _get_shape(batch, t_out, t_in):
        return (2, t_out, t_in)

    @property
    def tau_in(self):
        return GTau([part.shape[1] for part in self])

    @property
    def tau_out(self):
        return GTau([part.shape[0] for part in self])

    tau = tau_out

    def check_data(self, data):
        if any(part.numel() == 0 for part in data.values()):
            raise NotImplementedError('Non-zero parts in GWeights not currrently enabled!')

        shapes = set(part.shape for part in data.values())
        shape = shapes.pop()

        if not shape[self.zdim] == 2:
            raise ValueError('Complex dimension (dim={}) of each tensor should have length 2! Found: {}'.format(self.zdim, shape[self.zdim]))

    def as_parameter(self):
        """
        Return the weight as a :obj:`ParameterList` of :obj:`Parameter` so
        the weights can be added as parameters to a :obj:`torch.Module`.
        """
        return ParameterDictNew({key: Parameter(weight) for key, weight in self.items()})

    def rand(tau_in, tau_out, device=None, dtype=None, requires_grad=False):
        """
        Factory method to create a new random :obj:`GWeight`.
        """
        assert tau_out.keys() <= tau_in.keys(), "Tau after mixing can't include more irreps than before!"

        return GWeight({key: torch.rand((2, tau_out[key], tau_in[key]), device=device, dtype=dtype,
                          requires_grad=requires_grad) for key in tau_out.keys()})

    @staticmethod
    def randn(tau_in, tau_out, device=None, dtype=None, requires_grad=False):
        """
        Factory method to create a new random-normal :obj:`GWeight`.
        """
       
        assert tau_out.keys() <= tau_in.keys(), "Tau after mixing can't include more irreps than before!"

        return GWeight({key: torch.randn((2, tau_out[key], tau_in[key]), device=device, dtype=dtype,
                          requires_grad=requires_grad) for key in tau_out.keys()})

    @staticmethod
    def zeros(tau_in, tau_out, device=None, dtype=None, requires_grad=False):
        """
        Factory method to create a new random :obj:`GWeight`.
        """

        assert tau_out.keys() <= tau_in.keys(), "Tau after mixing can't include more irreps than before!"

        return GWeight({key: torch.zeros((2, tau_out[key], tau_in[key]), device=device, dtype=dtype,
                          requires_grad=requires_grad) for key in tau_out.keys()})

    @staticmethod
    def ones(tau_in, tau_out, device=None, dtype=None, requires_grad=False):
        """
        Factory method to create a new all-ones :obj:`GWeight`.
        """

        assert tau_out.keys() <= tau_in.keys(), "Tau after mixing can't include more irreps than before!"

        return GWeight({key: torch.ones((2, tau_out[key], tau_in[key]), device=device, dtype=dtype,
                          requires_grad=requires_grad) for key in tau_out.keys()})
