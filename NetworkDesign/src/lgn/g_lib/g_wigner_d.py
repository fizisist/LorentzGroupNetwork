import torch
from numpy import pi

# Hack to avoid circular imports
from lgn.g_lib import g_tau, g_tensor
from lgn.g_lib import rotations as rot

GTau = g_tau.GTau
GTensor = g_tensor.GTensor


class GWignerD(GTensor):
    """
    Core class for creating and tracking WignerD matrices.

    At the core of each :obj:`GWignerD` is a list of :obj:`torch.Tensors` with
    shape `(2*l+1, 2*l+1, 2)`, where:

    * `2*l+1` is the size of an irrep of weight `l`.
    * `2` corresponds to the real/imaginary parts of the complex dimension.

    Note
    ----

    For now, there is no batch or channel dimensions included. Although a
    G covariant network architecture with Wigner-D matrices is possible,
    the current scheme using PyTorch built-ins would be too slow to implement.
    A custom CUDA kernel would likely be necessary, and is a work in progress.

    Warning
    -------
    The constructor __init__() does not check that the tensor is actually
    a Wigner-D matrix, (that is an irreducible representation of the group G)
    so it is important to ensure that the input tensor is generated appropraitely.

    Parameters
    ----------

    data : iterable of of `torch.Tensor` with appropriate shape
        Input of a G vector.
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
    def rdim1(self):
        return 1

    @property
    def rdim2(self):
        return 2

    rdim = rdim2

    @property
    def keys(self):
        return self.keys()

    @staticmethod
    def _get_shape(batch, key, channels):
        return (2, (key[0]+1)*(key[1]+1), (key[0]+1)*(key[1]+1))

    def check_data(self, data):
        if any(part.numel() == 0 for part in data.values()):
            raise NotImplementedError('Non-zero parts in GWignerD not currrently enabled!')

        shapes = {key: part.shape for key, part in data.items()}

        zdims = {key: shape[self.zdim] for key, shape in shapes.items()}
        rdims = {key: (shape[self.rdim1], shape[self.rdim2]) for key, shape in shapes.items()}

        if not all(rdims[key][0] == (key[0]+1)*(key[1]+1) and rdims[key][1] == (key[0]+1)*(key[1]+1) for key in data.keys()):
            raise ValueError('Irrep dimension (dim={}) of each tensor should have shape 2*l+1! Found: {}'.format(self.rdim, rdims))

        if not all(zdim == 2 for zdim in zdims.values()):
            raise ValueError('Complex dimension (dim={}) of each tensor should have length 2! Found: {}'.format(self.zdim, zdims))

    @staticmethod
    def _bin_op_type_check(type1, type2):
        if type1 == GWignerD and type2 == GWignerD:
            raise ValueError('Cannot multiply two GWignerD!')

    @staticmethod
    def euler(maxdim, angles=None, device=None, dtype=None, requires_grad=False):
        """
        Factory method to create a new :obj:`GWeight`.

        If `angles=None`, will generate a uniformly distributed random Euler
        angle and then instantiate a GWignerD accordingly.
        """

        if angles is None:
            alpha, beta, gamma = torch.rand(3) * 2 * pi + 1j * torch.rand(3) * 2 * pi
            beta = beta / 2

        wigner_d = {(k,n): rot.LorentzD((k,n), alpha, beta, gamma, device=device, dtype=dtype) for k in range(maxdim) for n in range(maxdim)}

        return GWignerD(wigner_d)

    @staticmethod
    def rand(maxdim, device=None, dtype=None, requires_grad=False):
        """ Overwrite factor method inherited from :obj:`GTensor` since
        it would break covariance """
        raise NotImplementedError('Does not make sense as it would break covariance!')

    @staticmethod
    def randn(maxdim, device=None, dtype=None, requires_grad=False):
        """ Overwrite factor method inherited from :obj:`GTensor` since
        it would break covariance """
        raise NotImplementedError('Does not make sense as it would break covariance!')

    @staticmethod
    def zeros(maxdim, device=None, dtype=None, requires_grad=False):
        """ Overwrite factor method inherited from :obj:`GTensor` since
        it would break covariance """
        raise NotImplementedError('Does not make sense as it would break covariance!')

    @staticmethod
    def ones(maxdim, device=None, dtype=None, requires_grad=False):
        """ Overwrite factor method inherited from :obj:`GTensor` since
        it would break covariance """
        raise NotImplementedError('Does not make sense as it would break covariance!')
