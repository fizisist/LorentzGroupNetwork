import torch
from torch.nn import Module
from functools import reduce
from lgn.cg_lib import CGModule
from lgn.g_lib import g_torch, GWeight, GTau, GScalar


class MixReps(CGModule):
    """
    Module to linearly mix a representation from an input type `tau_in` to an
    output type `tau_out`.

    Input must have pre-defined types `tau_in` and `tau_out`.

    Parameters
    ----------
    tau_in : :obj:`GTau` (or compatible object).
        Input tau of representation.
    tau_out : :obj:`GTau` (or compatible object), or :obj:`int`.
        Input tau of representation. If an :obj:`int` is input,
        the output type will be set to `tau_out` for each
        parameter in the network.
    real : :obj:`bool`, optional
        Use purely real mixing weights.
    weight_init : :obj:`str`, optional
        String to set type of weight initialization.
    gain : :obj:`float`, optional
        Gain to scale initialized weights to.

    device : :obj:`torch.device`, optional
        Device to initialize weights to.
    dtype : :obj:`torch.dtype`, optional
        Data type to initialize weights to.
    """
    def __init__(self, tau_in, tau_out, real=False, weight_init='randn', gain=1, device=None, dtype=None):
        super().__init__(device=device, dtype=dtype)
        tau_in = GTau(tau_in)

        #remove any irreps with zero multiplicity
        tau_in = {key: val for key, val in tau_in.items() if val}
        # Allow one to set the output tau to a pre-specified number of output channels.
        tau_out = GTau(tau_out) if type(tau_out) is not int else tau_out
        if type(tau_out) is int:
            tau_out = {key: tau_out for key, val in tau_in.items() if val}

        self.tau_in = GTau(tau_in)
        self.tau_out = GTau(tau_out)
        self.real = real

        if weight_init == 'randn':
            weights = GWeight.randn(self.tau_in, self.tau_out, device=device, dtype=dtype)
        elif weight_init == 'rand':
            weights = GWeight.rand(self.tau_in, self.tau_out, device=device, dtype=dtype)
            weights = 2 * weights - 1
        else:
            raise NotImplementedError('weight_init can only be randn or rand for now')

        #multiply by gain
        gain = GScalar({key: torch.tensor([gain / max(shape) / (10 ** key[0] if key[0] == key[1] else 1), 0], device=device, dtype=dtype).view([2,1,1]) for key, shape in weights.shapes.items()})
        weights = gain * weights

        self.weights = weights.as_parameter()

    def forward(self, rep):
        """
        Linearly mix a represention.

        Parameters
        ----------
        rep : :obj:`list` of :obj:`torch.Tensor`
            Representation to mix.

        Returns
        -------
        rep : :obj:`list` of :obj:`torch.Tensor`
            Mixed representation.
        """
        if GTau.from_rep(rep) != self.tau_in:
            raise ValueError('Tau of input rep does not match initialized tau!'
                             ' rep: {} tau: {}'.format(GTau.from_rep(rep), self.tau_in))

        return g_torch.mix(self.weights, rep)

    @property
    def tau(self):
        return self.tau_out


class CatReps(Module):
    """
    Module to concanteate a list of reps. Specify input type for error checking
    and to allow network to fit into main architecture.

    Parameters
    ----------
    taus_in : :obj:`list` of :obj:`GTau` or compatible.
        List of taus of input reps.
    maxdim : :obj:`bool`, optional
        Maximum weight to include in concatenation.
    """
    def __init__(self, taus_in, maxdim=None):
        super().__init__()

        self.taus_in = taus_in = [GTau(tau) for tau in taus_in if tau]

        if maxdim is None:
            maxdim = max(sum(dict(i for tau in taus_in for i in tau.items()), ())) + 1
        self.maxdim = maxdim

        self.taus_in = taus_in
        self.tau_out = {}
        for tau in taus_in:
            for key, val in tau.items():
                if val > 0:
                    if max(key) <= maxdim - 1:
                        self.tau_out.setdefault(key, 0)
                        self.tau_out[key] += val
        self.tau_out = GTau(self.tau_out)

        self.all_keys = list(self.tau_out.keys())

    def forward(self, reps):
        """
        Concatenate a list of reps

        Parameters
        ----------
        reps : :obj:`list` of :obj:`GTensor` subclasses
            List of representations to concatenate.

        Returns
        -------
        reps_cat : :obj:`list` of :obj:`torch.Tensor`
        """
        # Drop Nones
        reps = [rep for rep in reps if rep is not None]

        # Error checking
        reps_taus_in = [rep.tau for rep in reps]
        if reps_taus_in != self.taus_in:
            raise ValueError('Tau of input reps does not match predefined version!'
                             'got: {} expected: {}'.format(reps_taus_in, self.taus_in))

        if self.maxdim is not None:
            reps = [rep.truncate(self.maxdim) for rep in reps]

        return g_torch.cat(reps)

    @property
    def tau(self):
        return self.tau_out


class CatMixReps(CGModule):
    """
    Module to concatenate mix a list of representation representations using
    :obj:`lgn.nn.CatReps`, and then linearly mix them using
    :obj:`lgn.nn.MixReps`.

    Parameters
    ----------
    taus_in : List of :obj:`GTau` (or compatible object).
        List of input tau of representation.
    tau_out : :obj:`GTau` (or compatible object), or :obj:`int`.
        Input tau of representation. If an :obj:`int` is input,
        the output type will be set to `tau_out` for each
        parameter in the network.
    maxdim : :obj:`bool`, optional
        Maximum weight to include in concatenation.
    real : :obj:`bool`, optional
        Use purely real mixing weights.
    weight_init : :obj:`str`, optional
        String to set type of weight initialization.
    gain : :obj:`float`, optional
        Gain to scale initialized weights to.

    device : :obj:`torch.device`, optional
        Device to initialize weights to.
    dtype : :obj:`torch.dtype`, optional
        Data type to initialize weights to.

    """
    def __init__(self, taus_in, tau_out, maxdim=None,
                 real=False, weight_init='randn', gain=1,
                 device=None, dtype=None):
        super().__init__(device=device, dtype=dtype)

        self.cat_reps = CatReps(taus_in, maxdim=maxdim)
        self.mix_reps = MixReps(self.cat_reps.tau, tau_out,
                                real=real, weight_init=weight_init, gain=gain,
                                device=device, dtype=dtype)

        self.taus_in = taus_in
        self.tau_out = GTau(self.mix_reps.tau)

    def forward(self, reps_in):
        """
        Concatenate and linearly mix a list of representations.

        Parameters
        ----------
        reps_in : :obj:`list` of :obj:`list` of :obj:`torch.Tensors`
            List of input representations.

        Returns
        -------
        reps_out : :obj:`list` of :obj:`torch.Tensors`
            Representation as a result of combining and mixing input reps.
        """
        reps_cat = self.cat_reps(reps_in)
        reps_out = self.mix_reps(reps_cat)

        return reps_out

    @property
    def tau(self):
        return self.tau_out
