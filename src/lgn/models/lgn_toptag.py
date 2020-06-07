import torch

import logging

from lgn.cg_lib import CGModule, ZonalFunctionsRel, ZonalFunctions, normsq4
from lgn.g_lib import GTau

from lgn.models.lgn_cg import LGNCG

from lgn.nn import RadialFilters
from lgn.nn import InputLinear, MixReps
from lgn.nn import OutputLinear, OutputPMLP, GetScalarsAtom
from lgn.nn import NoLayer

class LGNTopTag(CGModule):
    """
    Basic LGN Network used to train MD17 results in LGN paper.

    Parameters
    ----------
    maxdim : :obj:`int` of :class:`list` of :class:`int`
        Maximum weight in the output of CG products. (Expanded to list of
        length :obj:`num_cg_levels`)
    max_zf : :class:`int` of :class:`list` of :class:`int`
        Maximum weight in the output of the spherical harmonics  (Expanded to list of
        length :obj:`num_cg_levels`)
    num_cg_levels : :class:`int`
        Number of cg levels to use.
    num_channels : :class:`int` of :class:`list` of :class:`int`
        Number of channels that the output of each CG are mixed to (Expanded to list of
        length :obj:`num_cg_levels`)
    num_species : :class:`int`
        Number of species of atoms included in the input dataset.
    device : :class:`torch.device`
        Device to initialize the level to
    dtype : :class:`torch.torch.dtype`
        Data type to initialize the level to level to
    cg_dict : :class:`CGDict <lgn.cg_lib.CGDict>`
        Clebsch-gordan dictionary object.
    """
    def __init__(self, maxdim, max_zf, num_cg_levels, num_channels,
                 cutoff_type, hard_cut_rad, soft_cut_rad, soft_cut_width,
                 weight_init, level_gain, num_basis_fn,
                 top, input, num_mpnn_layers, activation='leakyrelu', pmu_in=False, add_beams=True,
                 scale=1, full_scalars=False, mlp=True, mlp_depth=None, mlp_width=None,
                 device=torch.device('cpu'), dtype=None, cg_dict=None):

        logging.info('Initializing network!')
        level_gain = expand_var_list(level_gain, num_cg_levels)

        hard_cut_rad = expand_var_list(hard_cut_rad, num_cg_levels)
        soft_cut_rad = expand_var_list(soft_cut_rad, num_cg_levels)
        soft_cut_width = expand_var_list(soft_cut_width, num_cg_levels)

        maxdim = expand_var_list(maxdim, num_cg_levels)
        max_zf = expand_var_list(max_zf, num_cg_levels)
        num_channels = expand_var_list(num_channels, num_cg_levels + 1)

        logging.info('hard_cut_rad: {}'.format(hard_cut_rad))
        logging.info('soft_cut_rad: {}'.format(soft_cut_rad))
        logging.info('soft_cut_width: {}'.format(soft_cut_width))
        logging.info('maxdim: {}'.format(maxdim))
        logging.info('max_zf: {}'.format(max_zf))
        logging.info('num_channels: {}'.format(num_channels))

        super().__init__(maxdim=max(maxdim + max_zf), device=device, dtype=dtype, cg_dict=cg_dict)
        device, dtype, cg_dict = self.device, self.dtype, self.cg_dict

        logging.info('CGdict maxdim: {}'.format(cg_dict.maxdim))

        self.num_cg_levels = num_cg_levels
        self.num_channels = num_channels
        self.scale = scale
        self.full_scalars = full_scalars
        self.pmu_in = pmu_in

        # Set up spherical harmonics
        if pmu_in:
            self.zonal_fns_in = ZonalFunctions(max(max_zf), device=device, dtype=dtype, cg_dict=cg_dict)
        self.zonal_fns = ZonalFunctionsRel(max(max_zf), device=device, dtype=dtype, cg_dict=cg_dict)

        # Set up position functions, now independent of spherical harmonics
        self.rad_funcs = RadialFilters(max_zf, num_basis_fn, num_channels, num_cg_levels, device=self.device, dtype=self.dtype)
        tau_pos = self.rad_funcs.tau

        if num_cg_levels:
            if add_beams:
                num_scalars_in = 2
            else:
                num_scalars_in = 1
        else:
            num_scalars_in = 202  # the second number should match the number of atoms (including beams)

        num_scalars_out = num_channels[0]

        if not pmu_in:
            self.input_func_atom = InputLinear(num_scalars_in, num_scalars_out,
                                               device=self.device, dtype=self.dtype)
        else:
            self.input_func_atom = MixReps(GTau({**{(0,0): num_scalars_in},**{(l,l): 1 for l in range(1, max_zf[0] + 1)}}), 
                                           GTau({(l,l): num_scalars_out for l in range(max_zf[0] + 1)}),
                                           device=self.device, dtype=self.dtype)
        
        tau_in_atom = self.input_func_atom.tau

        self.lgn_cg = LGNCG(maxdim, max_zf, tau_in_atom,
                                        tau_pos, num_cg_levels, num_channels,
                                        level_gain, weight_init, cutoff_type,
                                        hard_cut_rad, soft_cut_rad, soft_cut_width,
                                        mlp=mlp, mlp_depth=mlp_depth, mlp_width=mlp_width,
                                        activation=activation, device=self.device, dtype=self.dtype, cg_dict=self.cg_dict)

        tau_cg_levels_atom = self.lgn_cg.tau_levels_atom

        self.get_scalars_atom = GetScalarsAtom(tau_cg_levels_atom,
                                               device=self.device, dtype=self.dtype)

        num_scalars_atom = self.get_scalars_atom.num_scalars

        if top.lower().startswith('lin'):
            self.output_layer_atom = OutputLinear(num_scalars_atom, bias=True,
                                                  device=self.device, dtype=self.dtype)
        elif top.lower().startswith('pmlp'):
            self.output_layer_atom = OutputPMLP(num_scalars_atom, num_mixed=mlp_width,
                                                  device=self.device, dtype=self.dtype)

        logging.info('Model initialized. Number of parameters: {}'.format(sum(p.nelement() for p in self.parameters())))

    def forward(self, data, covariance_test=False):
        """
        Runs a forward pass of the network.

        Parameters
        ----------
        data : :obj:`dict`
            Dictionary of data to pass to the network.
        covariance_test : :obj:`bool`, optional
            If true, returns all of the atom-level representations twice.

        Returns
        -------
        prediction : :obj:`torch.Tensor`
            The output of the layer
        """
        # Get and prepare the data
        atom_scalars, atom_mask, edge_mask, atom_ps = self.prepare_input(data, self.num_cg_levels)

        # Calculate spherical harmonics and radial functions
        if self.pmu_in:
            zonal_functions_in, _, _ = self.zonal_fns_in(atom_ps)
            zonal_functions_in[(0, 0)] = torch.stack([atom_scalars.unsqueeze(-1),torch.zeros_like(atom_scalars.unsqueeze(-1))])
        zonal_functions, norms, sq_norms = self.zonal_fns(atom_ps, atom_ps)

        # Prepare the input reps for both the atom and edge network
        if self.num_cg_levels > 0:
            rad_func_levels = self.rad_funcs(norms, edge_mask * (norms != 0).byte())
            if not self.pmu_in:
                atom_reps_in = self.input_func_atom(atom_scalars, atom_mask)
            else:
                atom_reps_in = self.input_func_atom(zonal_functions_in)
        else:
            rad_func_levels = []
            atom_reps_in = self.input_func_atom(atom_scalars, atom_mask)

        # edge_net_in = self.input_func_edge(atom_scalars, atom_mask, edge_scalars, edge_mask, norms, sq_norms)

        # Clebsch-Gordan layers central to the network
        atoms_all = self.lgn_cg(atom_reps_in, atom_mask, rad_func_levels, zonal_functions)
        # Construct scalars for network output
        atom_scalars = self.get_scalars_atom(atoms_all)
        # edge_scalars = self.get_scalars_edge(edges_all)
        # Prediction in this case will depend only on the atom_scalars. Can make
        # it more general here.
        prediction = self.output_layer_atom(atom_scalars, atom_mask)

        # Covariance test
        if covariance_test:
            return prediction, atoms_all
        else:
            return prediction

    def prepare_input(self, data, cg_levels=True):
        """
        Extracts input from data class

        Parameters
        ----------
        data : ?????
            Information on the state of the system.

        Returns
        -------
        atom_scalars : :obj:`torch.Tensor`
            Tensor of scalars for each atom.
        atom_mask : :obj:`torch.Tensor`
            Mask used for batching data.
        atom_ps: :obj:`torch.Tensor`
            Positions of the atoms
        edge_mask: :obj:`torch.Tensor`
            Mask used for batching data.
        """
        device, dtype = self.device, self.dtype

        atom_ps = data['Pmu'].to(device, dtype) * self.scale

        data['Pmu'].requires_grad_(True)
        atom_mask = data['atom_mask'].to(device, torch.uint8)
        edge_mask = data['edge_mask'].to(device, torch.uint8)

        scalars = torch.ones_like(atom_ps[:, :, 0]).unsqueeze(-1)
        scalars = normsq4(atom_ps).abs().sqrt().unsqueeze(-1)

        if 'scalars' in data.keys():
            scalars = torch.cat([scalars, data['scalars'].to(device, dtype)], dim=-1)

        if not cg_levels:
            scalars = torch.stack(tuple(scalars for _ in range(scalars.shape[-1])), -2)

        return scalars, atom_mask, edge_mask, atom_ps


def expand_var_list(var, num_cg_levels):
    if type(var) is list:
        var_list = var + (num_cg_levels - len(var)) * [var[-1]]
    elif type(var) in [float, int]:
        var_list = [var] * num_cg_levels
    else:
        raise ValueError('Incorrect type {}'.format(type(var)))
    return var_list
