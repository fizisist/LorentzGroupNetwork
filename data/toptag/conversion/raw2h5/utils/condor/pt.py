#  File: pt.py
#  Author: Jan Offermann
#  Date: 03/10/20.

import numpy as np
from numba import jit

@jit
def pt(momentum):
    return np.sqrt(np.dot(momentum[1:3],momentum[1:3]))
