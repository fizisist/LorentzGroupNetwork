import torch
import torch.nn as nn

from lgn.cg_lib import CGModule
from lgn.g_lib import GTau, GScalar


class BasicMLP(nn.Module):
    """
    Multilayer perceptron used in various locations.  Operates only on the last axis of the data.

    Parameters
    ----------
    num_in : int
        Number of input channels
    num_out : int
        Number of output channels
    num_hidden : int, optional
        Number of hidden layers.
    layer_width : int, optional
        Width of each hidden layer (number of channels).
    activation : string, optional
        Type of nonlinearity to use.
    device : :obj:`torch.device`, optional
        Device to initialize the level to
    dtype : :obj:`torch.dtype`, optional
        Data type to initialize the level to
    """

    def __init__(self, num_in, num_out, num_hidden=1, layer_width=256, activation='leakyrelu', device=torch.device('cpu'), dtype=torch.float):
        super(BasicMLP, self).__init__()

        self.num_in = num_in

        self.linear = nn.ModuleList()
        self.linear.append(nn.Linear(num_in, layer_width))
        for i in range(num_hidden - 1):
            self.linear.append(nn.Linear(layer_width, layer_width))
        self.linear.append(nn.Linear(layer_width, num_out))

        activation_fn = get_activation_fn(activation)

        self.activations = nn.ModuleList()
        for i in range(num_hidden):
            self.activations.append(activation_fn)

        self.zero = torch.tensor(0, device=device, dtype=dtype)

        self.to(device=device, dtype=dtype)

    def forward(self, x, mask=None):
        # Standard MLP. Loop over a linear layer followed by a non-linear activation
        for (lin, activation) in zip(self.linear, self.activations):
            x = activation(lin(x))

        # After last non-linearity, apply a final linear mixing layer
        x = self.linear[-1](x)

        # If mask is included, mask the output
        if mask is not None:
            x = torch.where(mask, x, self.zero)

        return x

    def scale_weights(self, scale):
        self.linear[-1].weight *= scale
        if self.linear[-1].bias is not None:
            self.linear[-1].bias *= scale


def get_activation_fn(activation):
    activation = activation.lower()
    if activation == 'leakyrelu':
        activation_fn = nn.LeakyReLU()
    elif activation == 'relu':
        activation_fn = nn.ReLU()
    elif activation == 'elu':
        activation_fn = nn.ELU()
    elif activation == 'sigmoid':
        activation_fn = nn.Sigmoid()
    elif activation == 'logsigmoid':
        activation_fn = nn.LogSigmoid()
    elif activation == 'atan':
        activation_fn = ATan()
    else:
        raise ValueError('Activation function {} not implemented!'.format(activation))
    return activation_fn


class ATan(torch.nn.Module):
   
    def forward(self, input):
        return torch.atan(input)