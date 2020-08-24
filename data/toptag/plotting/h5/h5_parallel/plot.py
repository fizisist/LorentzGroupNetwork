#  File: convert.py
#  Author: Jan Offermann
#  Date: 10/05/19.
#  Goal: Make plots from the converted toptag datset.

import sys, os, re, time, uuid, glob
from utils.split import split
import subprocess as sub

def main(args):
    file_dir = str(sys.argv[1]) # converted toptag h5 file directory
    files = glob.glob(file_dir + '/*.h5')
    ndivisions = int(sys.argv[2]) # number of Condor workers to use for plotting
    cut_type = ''
    do_split = 1
    do_condor = 1
    
    if(len(sys.argv) > 3): cut_type = str(sys.argv[3])
    if(len(sys.argv) > 4): do_split = int(sys.argv[4])
    if(len(sys.argv) > 5): do_condor = int(sys.argv[5])

    # Split the files to be converted into multiple
    # chunks, each in its own folder. These will
    # be passed to HTCondor in separate jobs.
    if(do_split == 1):
        for idx, file in enumerate(files):
            print('Splitting ' + file + '... (' + str(idx + 1) + '/' + str(len(files)) + ')')
            split(file, 'run', ndivisions)
    files = [x.split('/')[-1] for x in files] # remove folder since files are now copied locally
    files = ','.join(files)
    
    if(do_condor == 1):
        # Create the Condor submit file, based on
        # the template file.
        text = ''
        with open('template/plot_template.sub','r') as f:
            text = f.readlines()
        with open('plot.sub', 'w') as f:
            for line in text:
                line = line.replace('$CUT', cut_type)
                line = line.replace('$FILENAMES', files)
                line = line.replace('$NJOBS', str(ndivisions))
                f.write(line)
        
        # Submit the jobs to Condor.
        sub.check_call(['condor_submit','plot.sub'])
        print('Jobs submitted to Condor. Good luck!')

if __name__ == '__main__':
    main(sys.argv)
