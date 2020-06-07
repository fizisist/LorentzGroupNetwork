import torch
from itertools import zip_longest


class GTau():
    """
    Class for keeping track of multiplicity (number of channels) of a G
    vector.

    Parameters
    ----------
    tau : :class:`list` of :class:`int`, :class:`GTau`, or class with `.tau` property.
        Multiplicity of an G vector.
    """
    def __init__(self, tau):
        if type(tau) is dict:
            if not all(type(t) == int for t in tau.values()):
                raise ValueError('Input must be a dict of int values! {}, values:{}'.format(type(tau), [type(t) for t in tau.values()]))
            if not all(type(t) == tuple for t in tau.keys()):
                raise ValueError('Input must be a dict with tuple keys! {}, keys:{}'.format(type(tau), [type(t) for t in tau.keys()]))
        else:
            try:
                tau = tau.tau
            except AttributeError:
                raise AttributeError('Input is of type %s does not have a defined .tau property!' % type(tau))

        self._tau = dict(tau)

    @property
    def maxdim(self):
        return max(list(sum(self._tau.keys(), ()))) + 1

    def keys(self):
        return self._tau.keys()

    def values(self):
        return self._tau.values()

    def items(self):
        return self._tau.items()

    def __iter__(self):
        """
        Loop over GTau
        """
        for t in self._tau.items():
            yield t

    def __getitem__(self, key):
        """
        Get item of GTau.
        """
        return self._tau[key]

    def __len__(self):
        """
        Length of GTau
        """
        return len(self._tau)

    def __setitem__(self, key, val):
        """
        Set value of GTau
        """
        self._tau[key] = val

    def __eq__(self, other):
        """
        Check equality of two :math:`GVec` compatible objects.
        """
        if self.keys() != other.keys():
            return False

        return all(self[key] == other[key] for key in self.keys())

    @staticmethod
    def cat(tau_list):
        """
        Return the multiplicity :class:`GTau` corresponding to the concatenation
        (direct sum) of a list of objects of type :class:`GTensor`.

        Parameters
        ----------
        tau_list : :class:`list` of :class:`GTau` or :class:`list` of :class:`int`s
            List of multiplicites of input :class:`GTensor`

        Return
        ------

        tau : :class:`GTau`
            Output tau of direct sum of input :class:`GTensor`.
        """
        tau_out = {}
        for tau in tau_list:
            for key, val in tau.items():
                if val>0:
                    tau_out.setdefault(key, 0)
                    tau_out[key] += val
        return GTau(tau_out)

    def __and__(self, other):
        return GTau.cat([self, other])

    def __rand__(self, other):
        return GTau.cat([self, other])

    def __str__(self):
        return str(dict(self._tau))

    __repr__ = __str__

    def __add__(self, other):
        return GTau.cat([self, other])

    def __radd__(self, other):
        """
        Reverse add, includes type checker to deal with sum([])
        """
        if type(other) is int:
            return self
        return GTau.cat([other, self])

    @staticmethod
    def from_rep(rep):
        """
        Construct GTau object from a GVec representation.

        Parameters
        ----------
        rep : :obj:`GTensor` :obj:`list` of :obj:`torch.Tensors`
            Input representation.

        """
        from lgn.g_lib.g_tensor import GTensor

        if rep is None:
            return GTau([])

        if isinstance(rep, GTensor):
            return rep.tau

        if torch.is_tensor(rep):
            raise ValueError('Input not compatible with GTensor')
        elif type(rep) is dict and any(type(irrep) != torch.Tensor for irrep in rep.values()):
            raise ValueError('Input not compatible with GTensor')

        tau = {key: irrep.shape[-2] for key, irrep in rep.items()}

        return GTau(tau)

    @property
    def tau(self):
        return self._tau

    @property
    def channels(self):
        channels = set(self._tau.values())
        if len(channels) == 1:
            return channels.pop()
        else:
            return None
