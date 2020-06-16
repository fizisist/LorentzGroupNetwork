#
#  beam_adjust.py
#
#  Use this to adjust the proton beam mass.
#
#  Created by Jan Offermann on 1/17/20.
#

import sys, os, glob
import h5py as h5, numpy as np
from numba import jit

@jit
def Energy(mass, momentum):
    p2 = np.dot(momentum,momentum)
    return np.sqrt(mass * mass + p2)


def main(args):

    file_dir = str(sys.argv[1])
    mass = float(sys.argv[2])
    files = glob.glob(file_dir + '/*.h5')
    
    for file in files:
        
        f = h5.File(file,'r+')
        keys = list(f.keys())
        
        if ('scalars' not in keys):
            print('Warning: key \'scalars\' not in keys of ' + file)
            f.close()
            continue
        
        # Set the beam masses in the scalars column
        f['scalars'][:,0,0] = mass
        f['scalars'][:,1,0] = mass
        
        # Also fix the beam 4-momenta. Same for each entry -> simple global adjustment
        f['Pmu'][:,0,0] = Energy(mass, f['Pmu'][0,0,1:])
        f['Pmu'][:,1,0] = Energy(mass, f['Pmu'][0,1,1:])
        
        f.close()
    
    


if __name__ == '__main__':
    main(sys.argv)

