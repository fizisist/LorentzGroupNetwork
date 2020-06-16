#
# slice.py
#
#  Created by Jan Offermann on 03/11/20.
#

import sys, os, subprocess as sub
#import numpy as np

def main(args):
    
    min = 550
    max = 650
    step = 10
    
    script = 'pt_slice.py'
    path_to_files = sys.argv[1]
    
    for i in range(min,max,step):
        sub.check_call(['python',script,path_to_files,str(i),str(i+step)])
    return

if __name__ == '__main__':
    main(sys.argv)

