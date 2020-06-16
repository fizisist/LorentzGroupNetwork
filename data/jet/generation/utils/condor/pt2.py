
# pt2.py

# Calculate pT^2 and sort jet constituents by decreasing pT.

import numpy as np
from numba import jit

@jit
def Pt2(vec):
    prod0 = np.multiply(vec[:,0],vec[:,0])
    prod1 = np.multiply(vec[:,1],vec[:,1])
    return np.add(prod0, prod1)

@jit
def Pt2_sort(vec):
    prod0 = np.multiply(vec[:,0],vec[:,0])
    prod1 = np.multiply(vec[:,1],vec[:,1])
    pt2 = np.add(prod0, prod1)
    return np.argsort(-pt2)
