import torch
import torch.nn as nn

from math import pi

from lgn.g_lib import GTau, GScalar


class RadialFilters(nn.Module):
    """
    Generate a set of learnable scalar functions for the aggregation/point-wise
    convolution step.

    One set of radial filters is created for each irrep (l = 0, ..., max_zf).

    Parameters
    ----------
    max_zf : :class:`int`
        Maximum l to use for the spherical harmonics.
    basis_set : iterable of :class:`int`
        Parameters of basis set to use. See :class:`RadPolyTrig` for more details.
    num_channels_out : :class:`int`
        Number of output channels to mix the resulting function into if mix
        is set to True in RadPolyTrig
    num_levels : :class:`int`
        Number of CG levels in the LGN.
    """
    def __init__(self, max_zf, num_basis_fn, num_channels_out, num_levels, mix=True, device=None, dtype=torch.float):
        if device is None:
            device = torch.device('cpu')
        super(RadialFilters, self).__init__()

        self.num_levels = num_levels
        self.max_zf = max_zf

        rad_funcs = [RadPolyTrig(max_zf[level], num_basis_fn, num_channels_out[level], mix=mix, device=device, dtype=dtype) for level in range(self.num_levels)]
        self.rad_funcs = nn.ModuleList(rad_funcs)
        self.tau = [{(l, l): rad_func.radial_types[l - 1] for l in range(0, maxzf + 1)} for rad_func, maxzf in zip(self.rad_funcs, max_zf)]

        if len(self.tau) > 0:
            self.num_rad_channels = self.tau[0][(1, 1)]
        else:
            self.num_rad_channels = 0

        # Other things
        self.device = device
        self.dtype = dtype

        self.zero = torch.tensor(0, device=device, dtype=dtype)

    def forward(self, norms, base_mask):
        """
        Forward pass of the network.

        Parameters
        ----------
        norms : :class:`torch.Tensor`
            Pairwise distance matrix between atoms.
        base_mask : :class:`torch.Tensor`
            Masking tensor with 1s on locations that correspond to active edges
            and zero otherwise.

        Returns
        -------
        rad_func_vals :  list of :class:`RadPolyTrig`
            Values of the radial functions.
        """

        return [rad_func(norms, base_mask) for rad_func in self.rad_funcs]


class RadPolyTrig(nn.Module):
    """
    A variation/generalization of spherical bessel functions.
    Rather than than introducing the bessel functions explicitly we just write out a basis
    that can produce them. Then, when apply a weight mixing matrix to reduce the number of channels
    at the end.
    """
    def __init__(self, max_zf, num_basis_fn, num_channels, mix=True, device=None, dtype=torch.float):
        if device is None:
            device = torch.device('cpu')
        super(RadPolyTrig, self).__init__()


        self.max_zf = max_zf

        self.num_basis_fn = num_basis_fn
        self.num_channels = num_channels
        self.device = device

        # This instantiates a set of functions sin(2*pi*n*x/a), cos(2*pi*n*x/a) with a=1.
        self.basis_size = (1, 1, 1, 2 * num_basis_fn)

        self.a = torch.randn(self.basis_size).to(device=device, dtype=dtype)
        self.b = torch.randn(self.basis_size).to(device=device, dtype=dtype)
        self.c = torch.randn(self.basis_size).to(device=device, dtype=dtype)
        # self.d = torch.rand(self.basis_size).to(device=device, dtype=dtype)
        # self.e = torch.rand(self.basis_size).to(device=device, dtype=dtype)
        # self.f = torch.rand(self.basis_size).to(device=device, dtype=dtype)

        self.a = nn.Parameter(self.a)
        self.b = nn.Parameter(self.b)
        self.c = nn.Parameter(self.c)
        # self.d = nn.Parameter(self.d)
        # self.e = nn.Parameter(self.e)
        # self.f = nn.Parameter(self.f)

        # If desired, mix the radial components to a desired shape
        self.mix = mix
        if (mix == 'cplx') or (mix is True):
            self.linear = nn.ModuleList([nn.Linear(2 * self.num_basis_fn, 2 * self.num_channels).to(device=device, dtype=dtype) for _ in range(max_zf + 1)])
            self.radial_types = (num_channels,) * (max_zf)
        elif mix == 'real':
            self.linear = nn.ModuleList([nn.Linear(2 * self.num_basis_fn, self.num_channels).to(device=device, dtype=dtype) for _ in range(max_zf + 1)])
            self.radial_types = (num_channels,) * (max_zf)
        elif (mix == 'none') or (mix is False):
            self.linear = None
            self.radial_types = (self.num_basis_fn,) * (max_zf)
        else:
            raise ValueError('Can only specify mix = real, cplx, or none! {}'.format(mix))

        self.zero = torch.tensor(0, device=device, dtype=dtype)

    def forward(self, norms, edge_mask):
        # Shape to resize to at the end
        s = norms.shape

        # Mask and reshape
        edge_mask = (edge_mask.byte()).unsqueeze(-1)
        norms = norms.unsqueeze(-1)

        # # Get inverse powers
        # rad_powers = torch.stack([torch.where(edge_mask, norms.pow(-pow), self.zero) for pow in range(1,self.rpow+1)], dim=-1)

        # Calculate trig functions
        # rad_trig = torch.where(edge_mask, self.a*(norms+self.b).pow(-1)+10*(self.c*norms+self.d)*(norms.pow(2)+self.e*norms+self.f).pow(-1), self.zero).unsqueeze(-1)
        # rad_trig = torch.where(edge_mask, self.c * torch.exp(-self.a.abs() * (norms + self.b).pow(2)), self.zero).unsqueeze(-1)
        
        # Lorentzian-bell radial functions (appear to work best)
        rad_trig = torch.where(edge_mask, self.b * (torch.ones_like(self.b) + (self.c * norms).pow(2)).pow(-1) + self.a, self.zero).unsqueeze(-1)
        
        # rad_trig = torch.where(edge_mask, 10 * (self.a * norms + self.b) * (norms.pow(2) + self.c.pow(2)).pow(-1), self.zero).unsqueeze(-1)
        # rad_trig = torch.where(edge_mask, torch.ones((1, 1, 1, 2*self.num_basis_fn),device=self.device), self.zero).unsqueeze(-1)
        # Take the product of the radial powers and the trig components and reshape
        rad_prod = rad_trig.view(s + (1, 2 * self.num_basis_fn,))
        # print("a=", self.linear[0].weight)

        # Apply linear mixing function, if desired
        if self.mix == 'cplx' or (self.mix is True):
            if len(s) == 3:
                radial_functions = [linear(rad_prod).view(s + (self.num_channels, 2)).permute(4, 0, 1, 2, 3) for linear in self.linear]
            elif len(s) == 2:
                radial_functions = [linear(rad_prod).view(s + (self.num_channels, 2)).permute(3, 0, 1, 2) for linear in self.linear]
        elif self.mix == 'real':
            radial_functions = [linear(rad_prod).view(s + (self.num_channels,)) for linear in self.linear]
        elif (self.mix == 'none') or (self.mix is False):
            radial_functions = [rad_prod.view(s + (self.num_basis_fn, 2)).permute(4, 0, 1, 2, 3)] * (self.max_zf)

        return GScalar({(l, l): radial_function for l, radial_function in enumerate(radial_functions)})
