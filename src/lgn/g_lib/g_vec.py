# Hack to avoid circular imports
from lgn.g_lib import g_tau, g_tensor, g_torch
from lgn.g_lib import g_scalar, g_wigner_d

GTau = g_tau.GTau
GTensor = g_tensor.GTensor
GScalar = g_scalar.GScalar
GWignerD = g_wigner_d.GWignerD


class GVec(GTensor):
    """
    Core class for creating and tracking G Vectors (aka G representations).

    At the core of each :obj:`GVec` is a list of :obj:`torch.Tensors` with
    shape `(B, C, 2*l+1, 2)`, where:

    * `B` is some number of batch dimensions.
    * `C` is the channels/multiplicity (tau) of each irrep.
    * `2*l+1` is the size of an irrep of weight `l`.
    * `2` corresponds to the real/imaginary parts of the complex dimension.

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
        return slice(1, -2)

    @property
    def cdim(self):
        return -2

    @property
    def rdim(self):
        return -1

    @staticmethod
    def _get_shape(batch, key, channels):
        return (2,) + tuple(batch) + (channels, (key[0]+1)*(key[1]+1))

    def check_data(self, data):
        if any(part.numel() == 0 for part in data.values()):
            raise ValueError('batch  dimensions! {}'.format(part.shape[self.bdim] for part in data))

        shapes = {key: part.shape for key, part in data.items()}

        # cdims = [shape[self.cdim] for shape in shapes]
        rdims = {key: shape[self.rdim] for key, shape in shapes.items()}
        zdims = {key: shape[self.zdim] for key, shape in shapes.items()}

        if not all(rdim == (key[0]+1)*(key[1]+1) for key, rdim in rdims.items()):
            raise ValueError('Irrep dimension (dim={}) of each tensor should have shape (k+1)*(n+1)! Found: {}'.format(self.rdim, rdims))

        if not all(zdim == 2 for zdim in zdims.values()):
            raise ValueError('Complex dimension (dim={}) of each tensor should have length 2! Found: {}'.format(self.zdim, zdims))

    def apply_wigner(self, wigner_d, side='left'):
        """
        Apply a WignerD matrix to `self`

        Parameters
        ----------
        wigner_d : :class:`GWignerD`
            The Wigner D matrix rotation to apply to `self`
        dir : :obj:`str`
            The direction to apply the Wigner-D matrices. Options are left/right.

        Returns
        -------
        :class:`GVec`
            The current :class:`GVec` rotated by :class:`GVec`
        """

        return g_torch.apply_wigner(wigner_d, self, side=side)
