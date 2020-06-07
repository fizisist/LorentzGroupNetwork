import torch

from abc import ABC, abstractmethod
from lgn.g_lib import g_torch, g_tau

GTau = g_tau.GTau


class GTensor(ABC):
    """
    Core class for creating and tracking G Vectors (aka G representations).

    Parameters
    ----------
    data : iterable of of `torch.Tensor` with appropriate shape
        Input of a G vector.
    """
    def __init__(self, data, ignore_check=False):
        if isinstance(data, type(self)):
            data = data.data

        data = {key: val for key, val in data.items() if (type(key) is tuple and torch.is_tensor(val) and val.numel() > 0)}

        if not ignore_check:
            self.check_data(data)

        self._data = data

    @abstractmethod
    def check_data(self, data):
        """
        Implement a data checking method.
        """
        if type(data) is not dict:
            raise ValueError('data must be a dictionary!')
        if not (type(key) is tuple for key in data.keys()):
            raise ValueError('Keys in data must be tuples of ints! {}'.format(data.keys()))
        if not all(torch.is_tensor(part) for part in data.values()):
            raise ValueError('Values in data must be torch tensors!'.format({key: type(val) for key, val in data.items()}))
        if not all(val.shape[-1] == (key[0]+1)*(key[1]+1) for key, val in data):
            raise ValueError('The last dimension of each torch tensor in data must match the dimension of the irrep labeled by the key, that is, dim((k,n))=(k+1)*(n+1)!')


    @property
    @abstractmethod
    def zdim(self):
        """
        Define the tau (channels) dimension for each part.
        """
        pass

    @property
    @abstractmethod
    def bdim(self):
        """
        Define the batch dimension for each part.
        """
        pass

    @property
    @abstractmethod
    def cdim(self):
        """
        Define the tau (channels) dimension for each part.
        """
        pass

    @property
    @abstractmethod
    def rdim(self):
        """
        Define the representation (2*l+1) dimension for each part. Should be None
        if it is not applicable for this type of GTensor.
        """
        pass

    @staticmethod
    @abstractmethod
    def _get_shape(batch, weight, channels):
        """
        Generate the shape of part based upon a batch size, multiplicity/number
        of channels, and weight.
        """
        pass

    def __len__(self):
        """
        Length of GVec.
        """
        return len(self._data)

    @property
    def maxdim(self):
        """
        Maximum weight (maxdim) of G object.

        Returns
        -------
        int
        """
        return max(list(sum(self._data.keys(), ()))) + 1

    def truncate(self, maxdim):
        """
        Update the maximum weight (`maxdim`) by truncating parts of the
        :class:`GTensor` if they correspond to weights greater than `maxdim`.

        Parameters
        ----------
        maxdim : :obj:`int`
            Maximum weight to truncate the representation to.

        Returns
        -------
        :class:`GTensor` subclass
            Truncated :class:`GTensor`
        """
        return self.__class__({key: val for key, val in self if max(key) < maxdim})

    @property
    def tau(self):
        """
        Multiplicity of each weight in G object.

        Returns
        -------
        :obj:`GTau`
        """
        return GTau({key: part.shape[self.cdim] for key, part in self})

    @property
    def bshape(self):
        """
        Get a list of shapes of each :obj:`torch.Tensor`
        """
        bshapes = {key: p.shape[self.bdim] for key, p in self}

        if len(set(bshapes)) != 1:
            raise ValueError('Every part must have the same shape! {}'.format(bshapes))

        return bshapes

    @property
    def shapes(self):
        """
        Get a list of shapes of each :obj:`torch.Tensor`
        """
        return {key: p.shape for key, p in self.items()}

    @property
    def channels(self):
        """
        Constructs :obj:`GTau`, and then gets the corresponding `GTau.channels`
        method.
        """
        return self.tau.channels

    @property
    def device(self):
        if any(list(self._data.values())[0].device != part.device for part in self._data.values()):
            raise ValueError('Not all parts on same device!')

        return list(self._data.values())[0].device

    @property
    def dtype(self):
        if any(list(self._data.values())[0].dtype != part.dtype for part in self._data.values()):
            raise ValueError('Not all parts using same data type!')

        return list(self._data.values())[0].dtype

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def pop(self, key):
        return self._data.pop(key)

    def __hash__(self):
        return hash((self.name, self.location))

    def __iter__(self):
        """
        Loop over GVec
        """
        for t in self._data.items():
            yield t

    def __getitem__(self, key):
        """
        Get item of GVec.
        """
        if type(key) is tuple:
            return self._data[key]
        else:
            raise ValueError('Keys of G tensors must be tuples of ints! {}'.format(key))

    def __setitem__(self, key, val):
        """
        Set index of GVec.
        """
        self._data[key] = val

    def __eq__(self, other):
        """
        Check equality of two :math:`GVec` compatible objects.
        """
        if self.keys() != other.keys():
            return False
        return all((self[key] == other[key]).all() for key in self.keys())

    @staticmethod
    def allclose(rep1, rep2, **kwargs):
        """
        Check equality of two :obj:`GTensor` compatible objects.
        """
        if len(rep1) != len(rep2):
            raise ValueError('')
        return all(torch.allclose(part1, part2, **kwargs) for part1, part2 in zip(rep1, rep2))

    def __and__(self, other):
        return self.cat([self, other])

    def __rand__(self, other):
        return self.cat([other, self])

    def __str__(self):
        return str(dict(self._data))

    __datar__ = __str__

    @classmethod
    def requires_grad(cls):
        return cls([t.requires_grad() for t in self._data.values()])

    def requires_grad_(self, requires_grad=True):
        self._data = {key: t.requires_grad_(requires_grad) for key, t in self._data.items()}
        return self

    def to(self, *args, **kwargs):
        self._data = {key: t.to(*args, **kwargs) for key, t in self._data.items()}
        return self

    def cpu(self):
        self._data = {key: t.cpu() for key, t in self._data.items()}
        return self

    def cuda(self, **kwargs):
        self._data = {key: t.cuda(**kwargs) for key, t in self._data.items()}
        return self

    def long(self):
        self._data = {key: t.long() for key, t in self._data.items()}
        return self

    def byte(self):
        self._data = {key: t.byte() for key, t in self._data.items()}
        return self

    def bool(self):
        self._data = {key: t.bool() for key, t in self._data.items()}
        return self

    def half(self):
        self._data = {key: t.half() for key, t in self._data.items()}
        return self

    def float(self):
        self._data = {key: t.float() for key, t in self._data.items()}
        return self

    def double(self):
        self._data = {key: t.double() for key, t in self._data.items()}
        return self

    def clone(self):
        return type(self)({key: t.clone() for key, t in self._data.items()})

    def detach(self):
        return type(self)({key: t.detach() for key, t in self._data.items()})

    @property
    def data(self):
        return self._data

    @property
    def grad(self):
        return type(self)({key: t.grad for key, t in self._data.items()})

    def add(self, other):
        return g_torch.add(self, other)

    def __add__(self, other):
        """
        Add element wise `torch.Tensors`
        """
        return g_torch.add(self, other)

    __radd__ = __add__

    def sub(self, other):
        return g_torch.sub(self, other)

    def __sub__(self, other):
        """
        Subtract element wise `torch.Tensors`
        """
        return g_torch.sub(self, other)

    __rsub__ = __sub__

    def mul(self, other):
        return g_torch.mul(self, other)

    def complex_mul(self, other):
        return g_torch.mul(self, other)

    def __mul__(self, other):
        """
        Add element wise `torch.Tensors`
        """
        return g_torch.mul(self, other)

    __rmul__ = __mul__

    def div(self, other):
        return g_torch.div(self, other)

    def __truediv__(self, other):
        """
        Add element wise `torch.Tensors`
        """
        return g_torch.div(self, other)

    __rtruediv__ = __truediv__

    def abs(self):
        """
        Calculate the element-wise absolute value of the :class:`torch.GTensor`.

        Warning
        -------
        Will break covariance!
        """

        return type(self)({key: part.abs() for key, part in self.items()})

    __abs__ = abs

    def max(self):
        """
        Returns a list of maximum values of each part in the
        :class:`torch.GTensor`.
        """

        return {key: part.max() for key, part in self.items()}

    def min(self):
        """
        Returns a list of minimum values of each part in the
        :class:`torch.GTensor`.
        """

        return {key: part.min() for key, part in self.items()}

            
    def squeeze(self, dim):
        return type(self)({key: part.squeeze(dim) for key, part in self.items()})

    def unsqueeze(self, dim):
        return type(self)({key: part.unsqueeze(dim) for key, part in self.items()})


    @classmethod
    def rand(cls, batch, tau, device=None, dtype=None, requires_grad=False, real=False):
        """
        Factory method to create a new random :obj:`GVec`.
        """

        shapes = {key: cls._get_shape(batch, key, t) for key, t in tau.items()}
        out = cls({key: torch.rand(shape, device=device, dtype=dtype, requires_grad=requires_grad) for key, shape in shapes.items()})
        if real:
            out = cls({key: torch.stack([t[0], 0 * t[0]]) for key, t in out.items()})
        return out

    @classmethod
    def randn(cls, tau, batch, device=None, dtype=None, requires_grad=False):
        """
        Factory method to create a new random :obj:`GVec`.
        """

        shapes = {key: cls._get_shape(batch, key, t) for key, t in tau.items()}

        return cls({key: torch.randn(shape, device=device, dtype=dtype,
                                requires_grad=requires_grad) for key, shape in shapes.items()})

    @classmethod
    def zeros(cls, tau, batch, device=None, dtype=None, requires_grad=False):
        """
        Factory method to create a new random :obj:`GVec`.
        """

        shapes = {key: cls._get_shape(batch, key, t) for key, t in tau.items()}

        return cls({key: torch.zeros(shape, device=device, dtype=dtype,
                                requires_grad=requires_grad) for key, shape in shapes.items()})

    @classmethod
    def ones(cls, tau, batch, device=None, dtype=None, requires_grad=False):
        """
        Factory method to create a new random :obj:`GVec`.
        """

        shapes = {key: cls._get_shape(batch, key, t) for key, t in tau.items()}

        return cls({key: torch.ones(shape, device=device, dtype=dtype,
                                requires_grad=requires_grad) for key, shape in shapes.items()})
