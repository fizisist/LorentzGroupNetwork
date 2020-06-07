import torch
import torch.nn as nn

from lgn.cg_lib import CGProduct
from lgn.nn.generic_levels import get_activation_fn
from lgn.nn import CatMixReps


class LGNAtomLevel(nn.Module):
    """
    Basic NBody level initialization.
    """
    def __init__(self, tau_in, tau_pos, maxdim, num_channels, level_gain, weight_init,
                 device=torch.device('cpu'), dtype=torch.float, cg_dict=None):
        super(LGNAtomLevel, self).__init__()

        self.maxdim = maxdim
        self.num_channels = num_channels

        self.tau_in = tau_in
        self.tau_pos = tau_pos

        # Operations linear in input reps
        self.cg_aggregate = CGProduct(tau_in, tau_pos, maxdim=self.maxdim, aggregate=True, device=device, dtype=dtype, cg_dict=cg_dict)
        tau_ag = self.cg_aggregate.tau_out
        self.cg_power = CGProduct(tau_in, tau_in, maxdim=self.maxdim, device=device, dtype=dtype, cg_dict=cg_dict)
        tau_sq = self.cg_power.tau_out

        self.cat_mix = CatMixReps([tau_ag, tau_in, tau_sq], num_channels, maxdim=self.maxdim, weight_init=weight_init, gain=level_gain, device=device, dtype=dtype)
        self.tau_out = self.cat_mix.tau_out

    def forward(self, atom_reps, edge_reps, mask):
        # Aggregate information based upon edge reps
        reps_ag = self.cg_aggregate(atom_reps, edge_reps)
        # CG non-linearity for each atom
        reps_sq = self.cg_power(atom_reps, atom_reps)
        # Concatenate and mix results
        reps_out = self.cat_mix([reps_ag, atom_reps, reps_sq])
        return reps_out

class CGMLP(nn.Module):
    """
    Multilayer perceptron acting on scalar parts of GVec's.  Operates only on the last axis of the data.

    """

    def __init__(self, tau, num_hidden=3, layer_width_mul=2, activation='sigmoid', device=torch.device('cpu'), dtype=torch.float):
        super(CGMLP, self).__init__()

        self.tau = tau
        num_scalars = 2 * tau[(0, 0)]
        self.num_scalars = num_scalars
        layer_width = layer_width_mul * num_scalars

        self.linear = nn.ModuleList()
        self.linear.append(nn.Linear(num_scalars, layer_width))
        for i in range(num_hidden - 1):
            self.linear.append(nn.Linear(layer_width, layer_width))
        if num_hidden > 0:
            self.linear.append(nn.Linear(layer_width, num_scalars))
        else:
            self.linear.append(nn.Linear(num_scalars, num_scalars))

        activation_fn = get_activation_fn(activation)

        self.activations = nn.ModuleList()
        for i in range(num_hidden):
            self.activations.append(activation_fn)

        self.zero = torch.tensor(0, device=device, dtype=dtype)

        self.to(device=device, dtype=dtype)

    def forward(self, reps_in, mask=None):
        # Standard MLP. Loop over a linear layer followed by a non-linear activation
        
        reps_out = reps_in
        x = reps_out.pop((0, 0)).squeeze(-1)
        s = x.shape

        x = x.permute(1,2,3,0).contiguous().view(s[1:3] + (self.num_scalars,))

        for (lin, activation) in zip(self.linear, self.activations):
            x = activation(lin(x))
        # After last non-linearity, apply a final linear mixing layer
        x = self.linear[-1](x)

        # If mask is included, mask the output
        if mask is not None:
            x = torch.where(mask, x, self.zero)

        reps_out[(0, 0)] = x.view(s[1:]+(2,)).permute(3,0,1,2).unsqueeze(-1)
        return reps_out

    def scale_weights(self, scale):
        self.linear[-1].weight *= scale
        if self.linear[-1].bias is not None:
            self.linear[-1].bias *= scale
