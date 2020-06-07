import torch
import torch.nn as nn

from lgn.nn import BasicMLP
from lgn.g_lib import cat
from lgn.cg_lib import normsq

############# Get Scalars #############

class GetScalarsAtom(nn.Module):
    r"""
    Construct a set of scalar feature vectors for each atom by using the
    covariant atom :class:`GVec` representations at various levels.

    Parameters
    ----------
    tau_levels : :class:`list` of :class:`GTau`
        Multiplicities of the output :class:`GVec` at each level.
    full_scalars : :class:`bool`, optional
        Construct a more complete set of scalar invariants from the full
        :class:`GVec` (``true``), or just use the :math:``\ell=0`` component
        (``false``).
    device : :class:`torch.device`, optional
        Device to instantite the module to.
    dtype : :class:`torch.dtype`, optional
        Data type to instantite the module to.
    """
    def __init__(self, tau_levels, full_scalars=True, device=torch.device('cpu'), dtype=torch.float):
        super().__init__()

        self.device = device
        self.dtype = dtype

        self.maxdim = max(len(tau) for tau in tau_levels) - 1

        split_l0 = [tau[(0, 0)] for tau in tau_levels]
        split_full = [sum(tau.values()) for tau in tau_levels]
        self.full_scalars = full_scalars
        if full_scalars:
            self.num_scalars = sum(split_full) # + sum(split_l0)
            self.split = split_full
        else:
            self.num_scalars = sum(split_l0)
            self.split = split_l0

    def forward(self, reps_all_levels):
        """
        Forward step for :class:`GetScalarsAtom`

        Parameters
        ----------
        reps_all_levels : :class:`list` of :class:`GVec`
            List of covariant atom features at each level

        Returns
        -------
        scalars : :class:`torch.Tensor`
            Invariant scalar atom features constructed from ``reps_all_levels``
        """

        reps = cat(reps_all_levels)
        scalars = reps.pop((0, 0))

        if self.full_scalars and len(reps.keys())>0:
            scalars_full = list(normsq(reps).values())
            scalars = [scalars] + scalars_full
            scalars = torch.cat(scalars, dim=reps.cdim)
        return scalars


############# Output of network #############

class OutputLinear(nn.Module):
    """
    Module to create prediction based upon a set of rotationally invariant
    atom feature vectors. This is performed in a permutation invariant way
    by using a (batch-masked) sum over all atoms, and then applying a
    linear mixing layer to predict a single output.

    Parameters
    ----------
    num_scalars : :class:`int`
        Number scalars that will be used in the prediction at the output
        of the network.
    bias : :class:`bool`, optional
        Include a bias term in the linear mixing level.
    device : :class:`torch.device`, optional
        Device to instantite the module to.
    dtype : :class:`torch.dtype`, optional
        Data type to instantite the module to.
    """
    def __init__(self, num_scalars, bias=True, device=None, dtype=torch.float):
        if device is None:
            device = torch.device('cpu')
        super(OutputLinear, self).__init__()

        self.num_scalars = num_scalars
        self.bias = bias
        if num_scalars > 0:
            self.lin = nn.Linear(2 * num_scalars, 2, bias=bias)
            self.lin.to(device=device, dtype=dtype)
        self.zero = torch.tensor(0, dtype=dtype, device=device)

    def forward(self, scalars, _):
        """
        Forward step for :class:`OutputLinear`

        Parameters
        ----------
        atom_scalars : :class:`torch.Tensor`
            Scalar features for each atom used to predict the final learning target.
        atom_mask : :class:`torch.Tensor`
            Unused. Included only for pedagogical purposes.

        Returns
        -------
        predict : :class:`torch.Tensor`
            Tensor used for predictions.
        """
        if self.num_scalars > 0:
            batch_size = scalars.shape[1]
            scalars = scalars.sum(2).permute(1, 2, 3, 0)  # sum over atoms to ensure permutation invariance
            scalars = scalars.contiguous().view((batch_size, -1))  # put the complex dimension at the end and collapse into one dimension of scalars
            predict = self.lin(scalars)  # .softmax(dim=-1) #apply linear mixing to scalars in each event
        else:
            predict = scalars
        return predict


class OutputPMLP(nn.Module):
    """
    Module to create prediction based upon a set of rotationally invariant
    atom feature vectors.

    This is peformed in a three-step process::

    (1) A MLP is applied to each set of scalar atom-features.
    (2) The environments are summed up.
    (3) Another MLP is applied to the output to predict a single learning target.

    Parameters
    ----------
    num_scalars : :class:`int`
        Number scalars that will be used in the prediction at the output
        of the network.
    bias : :class:`bool`, optional
        Include a bias term in the linear mixing level.
    device : :class:`torch.device`, optional
        Device to instantite the module to.
    dtype : :class:`torch.dtype`, optional
        Data type to instantite the module to.
    """
    def __init__(self, num_scalars, num_mixed=2, activation='leakyrelu', device=None, dtype=torch.float):
        if device is None:
            device = torch.device('cpu')
        super(OutputPMLP, self).__init__()

        self.num_scalars = num_scalars
        self.num_mixed = num_mixed

        self.mlp1 = BasicMLP(2*num_scalars, num_scalars * num_mixed, num_hidden=1, activation=activation, device=device, dtype=dtype)
        self.mlp2 = BasicMLP(num_scalars * num_mixed, 2, num_hidden=1, activation=activation, device=device, dtype=dtype)

        self.zero = torch.tensor(0, device=device, dtype=dtype)

    def forward(self, atom_scalars, atom_mask):
        """
        Forward step for :class:`OutputPMLP`

        Parameters
        ----------
        atom_scalars : :class:`torch.Tensor`
            Scalar features for each atom used to predict the final learning target.
        atom_mask : :class:`torch.Tensor`
            Unused. Included only for pedagogical purposes.

        Returns
        -------
        predict : :class:`torch.Tensor`
            Tensor used for predictions.
        """
        # Reshape scalars appropriately
        atom_scalars = atom_scalars.view(atom_scalars.permute(1,2,3,4,0).shape[:2] + (2*self.num_scalars,))

        # First MLP applied to each atom
        x = self.mlp1(atom_scalars)

        # Reshape to sum over each atom in molecules, setting non-existent atoms to zero.
        atom_mask = atom_mask.unsqueeze(-1)
        x = torch.where(atom_mask, x, self.zero).sum(1)

        # Prediction on permutation invariant representation of molecules
        predict = self.mlp2(x)

        # predict = predict.squeeze(-1)
        return predict
