import torch
import numpy as np
from math import sqrt
import lgn.cg_lib
from lgn.cg_lib import CGProduct, cg_product, CGDict, zonal_functions4
from lgn.g_lib import GVec, GTau, rotations
from lgn.models.autotest.lgn_tests import _gen_rot
device = torch.device('cpu')
dtype = torch.float

from colorama import Fore, Back, Style 


"""
Script checking covariance of the CGProduct of two GVec's. Can specify batch size and representation types of the vectors below. 
Simply run this file with no arguments to see the results.
"""


##### INPUTS

nbatch = 1       # number of batches (will be equal for both representations)
natoms = 1       # number of atoms in each batch (will be equal for both representations)
nchan = 1        # number of channels in each irrep of each representation (must be uniform because of our restrictions)
tau1 = GTau({(1, 1): nchan})  # the representation types of the two vectors to be multiplied
tau2 = GTau({(1, 1): nchan})
aggregate = True   # whether you want to aggregate. This creates rep1 of batch dimension (nbatch, natoms, natoms) and rep2 of (nbatch, natoms) and then sums over one pair of atom indices.

(alpha, beta, gamma) = (1+2j, 1+3j, 1+1j)  # Complex Euler angles to rotate by

accuracy = 1e-4     # absolute error up to which answers should match


############################################################## TEST

maxk1 = max({key[0] for key in tau1.keys()})
maxn1 = max({key[1] for key in tau1.keys()})
maxk2 = max({key[0] for key in tau2.keys()})
maxn2 = max({key[1] for key in tau2.keys()})
maxdim = max(maxk1+maxk2,maxn1+maxn2) + 1
cg_dict = CGDict(maxdim=maxdim)

print("Running covariance test with parameters:")
print("nbatch=", nbatch)
print("natoms=", natoms)
print("tau1=", tau1)
print("tau2=", tau2)
print("(alpha,beta,gamma)=", (alpha, beta, gamma))
print("accuracy=", accuracy, "\n--------------------------------")

cartesian4 = torch.tensor([[[1, 0, 0, 0], [0, 1 / sqrt(2.), 0, 0], [0, 0, 0, 1], [0, -1 / sqrt(2.), 0, 0]],
                               [[0, 0, 0, 0], [0, 0, -1 / sqrt(2.), 0], [0, 0, 0, 0], [0, 0, -1 / sqrt(2.), 0]]], device=device)
cartesian4H = torch.tensor([[[1, 0, 0, 0], [0, 1 / sqrt(2.), 0, 0], [0, 0, 0, 1], [0, -1 / sqrt(2.), 0, 0]],
                            [[0, 0, 0, 0], [0, 0, 1 / sqrt(2.), 0], [0, 0, 0, 0], [0, 0, 1 / sqrt(2.), 0]]], device=device).permute(0, 2, 1)

rand_p = lambda nbatch, natoms: GVec({(1, 1): torch.tensor([1., 0.]).view(2, 1, 1, 1) * torch.rand(2, nbatch, natoms, 4)}, ignore_check=True)
rand_rep = lambda nbatch, natoms, tau: GVec({(k, n): torch.rand(2, nbatch, natoms, t, (k + 1) * (n + 1)) for ((k, n), t) in tau.items()})
rand_rep_agg = lambda nbatch, natoms, tau: GVec({(k, n): torch.rand(2, nbatch, natoms, natoms, t, (k + 1) * (n + 1)) for ((k, n), t) in tau.items()})

# pos = rand_p(nbatch, natoms)
pos = GVec({(1, 1): torch.tensor([[[[0., 0., 1., 0.]]],
                                  [[[0., 0., 0., 0.]]]])}, ignore_check=True)
zf = zonal_functions4(cg_dict, pos[(1,1)][0], 1, normalize=False)[0]
D, R = _gen_rot((alpha, beta, gamma), maxdim, cg_dict=cg_dict)
I = rotations.LorentzD((0,0),0,0,0)
p_rotated = pos.apply_wigner({(0,0):I, (1,1): torch.stack([R,0. * R],0)}, side='left')
zf_p_rotated = zonal_functions4(cg_dict, p_rotated[(1, 1)][0], 1, normalize=False)[0]
zf_rotated = zf.apply_wigner(D, side='left')
deviation = {key: torch.abs(torch.add(zf_rotated[key], -zf_p_rotated[key])).max() for key in zf.keys()}
print("ZF DEVIATION: ", deviation)


if aggregate:
    rep1 = GVec({}).rand((nbatch, natoms, natoms), tau1)
else:
    rep1 = GVec({}).rand((nbatch, natoms), tau1)
rep2 = GVec({}).rand((nbatch, natoms), tau2)
print("ready: reps created")

#rep_product = CGProduct(tau1=tau1, tau2=tau2, maxdim=1, device=device, dtype=dtype)
print("ready: rep_product() initialized")

#rep_prod=rep_product(rep1,rep2)
rep_prod = cg_product(cg_dict, rep1, rep2, aggregate=aggregate)
print("ready: product of non-rotated vectors")

angles = (alpha, beta, gamma)
r1 = rotations.rotate_rep(rep1, *angles, cg_dict=cg_dict)
r2 = rotations.rotate_rep(rep2, *angles, cg_dict=cg_dict)
print("ready: rotated vectors")

#rep_tensor=rep_product(r1,r2)
rep_tensor = cg_product(cg_dict, r1, r2, aggregate=aggregate)
print("ready: product of rotated vectors")

rotated_tensor = rotations.rotate_rep(rep_prod, *angles, cg_dict=cg_dict)
print("ready: rotated product")

test = all([torch.all(torch.lt(torch.abs(torch.add(rep_tensor[key], -rotated_tensor[key])), accuracy)) for key in rep_tensor.keys()])
deviation = {key: torch.abs(torch.add(rep_tensor[key], -rotated_tensor[key])).max() for key in rep_tensor.keys()}

if test:
    print(Fore.GREEN+"PASSED covariance test up to ", accuracy)
else:
    print(Fore.RED+"FAILED covariance test to required precision")

print(Style.RESET_ALL)
print("Max deviations by irrep ", deviation)

print("For visual comparison, here is the tensor obtained by rotations and THEN product:")
print(rep_tensor)
print("Now the one obtained by FIRST multiplying and then rotating:")
print(rotated_tensor)
