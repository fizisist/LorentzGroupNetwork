# First import G-related modules and classes

# Import some basic complex utilities
from lgn.g_lib.cplx_lib import mul_zscalar_zirrep, mul_zscalar_zscalar
from lgn.g_lib.cplx_lib import mix_zweight_zvec, mix_zweight_zscalar

# This is necessary to avoid ImportErrors with circular dependencies
from lgn.g_lib import g_tau, g_torch, g_tensor
from lgn.g_lib import g_vec, g_scalar, g_weight, g_wigner_d
from lgn.g_lib import rotations

# Begin input of G-related utilities
from lgn.g_lib.g_tau import GTau
from lgn.g_lib.g_tensor import GTensor
from lgn.g_lib.g_wigner_d import GWignerD
from lgn.g_lib.g_vec import GVec
from lgn.g_lib.g_scalar import GScalar
from lgn.g_lib.g_weight import GWeight
from lgn.g_lib.parameter_dict_new import ParameterDictNew

# Network type structures
from lgn.g_lib.g_torch import cat, mix, cat_mix
