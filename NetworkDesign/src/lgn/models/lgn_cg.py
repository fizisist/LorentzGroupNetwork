import logging
import torch.nn as nn

from lgn.models import LGNAtomLevel, CGMLP
from lgn.cg_lib import CGModule

class LGNCG(CGModule):
    def __init__(self, maxdim, max_zf, tau_in_atom, tau_pos,
                 num_cg_levels, num_channels,
                 level_gain, weight_init,
                 cutoff_type, hard_cut_rad, soft_cut_rad, soft_cut_width,
                 mlp=True, mlp_depth=None, mlp_width=None, activation='leakyrelu',
                 device=None, dtype=None, cg_dict=None):
        super().__init__(device=device, dtype=dtype, cg_dict=cg_dict)
        device, dtype, cg_dict = self.device, self.dtype, self.cg_dict

        self.max_zf = max_zf
        self.mlp = mlp
        tau_atom_in = tau_in_atom.tau if type(tau_in_atom) is CGModule else tau_in_atom

        logging.info('{}'.format(tau_atom_in))

        atom_levels = nn.ModuleList()
        if mlp:
            mlp_levels = nn.ModuleList()

        tau_atom = tau_atom_in

        for level in range(num_cg_levels):
            # Now add the NBody level
            atom_lvl = LGNAtomLevel(tau_atom, tau_pos[level], maxdim[level], num_channels[level+1],
                                          level_gain[level], weight_init,
                                          device=device, dtype=dtype, cg_dict=cg_dict)
            atom_levels.append(atom_lvl)
            if mlp:
                mlp_lvl = CGMLP(atom_lvl.tau_out, activation=activation, num_hidden=mlp_depth, layer_width_mul=mlp_width, device=device, dtype=dtype)
                mlp_levels.append(mlp_lvl)
            tau_atom = atom_lvl.tau_out

            logging.info('{}'.format(tau_atom))

        self.atom_levels = atom_levels
        if mlp:
            self.mlp_levels = mlp_levels

        self.tau_levels_atom = [tau_atom_in] + [level.tau_out for level in atom_levels]

    def forward(self, atom_reps, atom_mask, rad_funcs, zonal_functions):
        """
        Runs a forward pass of the LGN CG layers.

        Parameters
        ----------
        atom_reps :  G Vector
            Input atom representations.
        atom_mask : :obj:`torch.Tensor` with data type `torch.byte`
            Batch mask for atom representations. Shape is
            :math:`(N_{batch}, N_{atom})`.
        edge_net : G Scalar or None`
            Input edge scalar features.
        edge_mask : :obj:`torch.Tensor` with data type `torch.byte`
            Batch mask for atom representations. Shape is
            :math:`(N_{batch}, N_{atom}, N_{atom})`.
        rad_funcs : :obj:`list` of G Scalars
            The (possibly learnable) radial filters.
        edge_mask : :obj:`torch.Tensor`
            Matrix of the magnitudes of relative position vectors of pairs of atoms.
            :math:`(N_{batch}, N_{atom}, N_{atom})`.
        zonal_functions : G Vector
            Representation of spherical harmonics calculated from the relative
            position vectors of pairs of points.

        Returns
        -------
        atoms_all : list of G Vectors
            The concatenated output of the representations output at each level.
        edges_all : list of G Scalars
            The concatenated output of the scalar edge network output at each level.
        """
        assert len(self.atom_levels) == len(rad_funcs)

        # Construct iterated multipoles
        atoms_all = [atom_reps]
        if self.mlp:
            for idx, (atom_level, mlp_level) in enumerate(zip(self.atom_levels, self.mlp_levels)):
                edge_reps = rad_funcs[idx] * zonal_functions
                atom_reps = atom_level(atom_reps, edge_reps, atom_mask)
                atom_reps = mlp_level(atom_reps)
                atoms_all.append(atom_reps)
        else:
            for idx, atom_level in enumerate(self.atom_levels):
                edge_reps = rad_funcs[idx] * zonal_functions
                atom_reps = atom_level(atom_reps, edge_reps, atom_mask)
                atoms_all.append(atom_reps)

        return atoms_all
