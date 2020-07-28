#  File: convert.py
#  Author: Jan Offermann
#  Date: 10/05/19.
#  Goal: Convert the top-tagging reference dataset to our own HDF5 format.

import sys, os, time
from utils.split import raw_split, ttv_split
from utils.check_jobs import check_jobs
from utils.display import display_time
from utils.concat_fancy import concatN, concatN_r, concatN_condor_r # TODO: Are these all needed?
import subprocess as sub
import numpy as np
    
def main(args):
    file_to_convert = str(sys.argv[1]) # raw toptag file
    file_to_convert_nopath = file_to_convert.split('/')[-1]
    ndivisions = int(sys.argv[2]) # number of Condor workers to use for conversion
    
    # add_beams: optional, determines whether or not to add beams to 'Pmu' column, and add a 'scalar' column with particle masses & labels
    if(len(sys.argv) > 3): add_beams = int(sys.argv[3])
    else: add_beams = 0
    
    # dot_products: optional, determines whether or not to add dot products of 4-momenta to a 'dots' column
    if(len(sys.argv) > 4): dot_products = int(sys.argv[4])
    else: dot_products = 0
    
    # double_precision: optional, determines whether to use double or float precision (default is 1 for double)
    if(len(sys.argv) > 5): double_precision = int(sys.argv[5])
    else: double_precision = 1
    
    # mode: optional, determines what concatenation strategy to use
    if(len(sys.argv) > 6): mode = int(sys.argv[6])
    else: mode = 2
    
    # If the user passes ndivisions = -1, we just perform conversion locally, *without* condor.
    # This may be slower (& memory-intensive, in the case of dot_products = 1), but necessary
    # if the user does not have condor set up.
    
    if(ndivisions = -1):
        print('Performing conversion locally (i.e. without using HTCondor for parallelization).')
        sub.check_call(['python3','utils/condor/raw2h5.py',file_to_convert,str(add_beams),str(dot_products),str(double_precision)])
        return
    
    timer_start = time.time()
    # Split the file to be converted into multiple
    # chunks, each in its own folder. These will
    # be passed to HTCondor in separate jobs.
    raw_split(file_to_convert, 'run', ndivisions, 'table')
    
    # Create the Condor submit file, based on
    # the template file.
    text = ''
    with open('template/convert_template.sub','r') as f:
        text = f.readlines()
    with open('raw2h5.sub', 'w') as f:
        for line in text:
            line = line.replace('$FILENAME', file_to_convert_nopath)
            line = line.replace('$ADD_BEAMS', str(add_beams))
            line = line.replace('$DOT_PRODUCTS', str(dot_products))
            line = line.replace('$NJOBS', str(ndivisions))
            line = line.replace('$PRECISION', str(double_precision))
            f.write(line)
        
    # Submit the jobs to Condor.
    sub.check_call(['condor_submit','raw2h5.sub'])
    print('Conversion jobs submitted to Condor. Good luck!')
    
    # If performing dot products, avoid concatenation -- in practice this is likely to cause memory errors on condor nodes,
    # since the file sizes will be considerably larger than normal. Thus we don't need to explicitly wait for job completion.
    if(dot_products): return #
    
    # Wait for the Condor jobs to finish & check that they did.
    # TODO: For now, we will set a maximum wait time. Should eventually check for crashes & maybe resubmit jobs that fail.
    
    status = False
#    max_wait_time = 21600 # seconds ( = 6 hours) TODO: May require tweaking. Generous for default conversion, but dot products may take a while?
    max_wait_time = np.maximum(int(86400 / ndivisions), 21600) # max wait time scales inversely with # of cores, minimum is 6 hours
    if(dot_products): max_wait_time = 5. * max_wait_time # rescale (somewhat arbitrarily) if we calculate dot products, these will take longer guaranteed
    print('(Maximum wait time for conversion jobs is ' + str(display_time(max_wait_time)) + ').')
    sleep_time = 15 # seconds
    start = time.time()
    while True:
        time.sleep(sleep_time) # sleep for "sleep_time" number of seconds
        [status,success] = check_jobs(ndivisions) # check log files of jobs
        if(status == True): break # if jobs are completed, exit loop
        now = time.time()
        if(now - start > max_wait_time): break # exit loop if too much time passed
    
    if (status == False):
        print('Uh oh, some jobs failed or didn\'t complete on-time.')
        return
    
    # Jobs are completed, concatenate results. (No TTV split for toptag, raw files are already split)
    print('Conversion jobs completed successfully. Preparing concatenation jobs.')
    
    file_list = ['run'+str(x)+'/'+file_to_convert_nopath.replace('.h5','_c.h5') for x in range(ndivisions)]

    # mode = 0: Simple, serialized. Slow but easy to follow.
    if(mode == 0):
        concatN([x for x in file_list], file_to_convert_nopath)
        
    # mode = 1: Pairwise, serialized & recursive. Faster than mode = 1, still slow vs. parallelization.
    elif (mode == 1):
        concatN_r([x for x in file_list], file_to_convert_nopath)

    elif (mode == 2):
        concatN_condor_r([x for x in file_list], file_to_convert_nopath)

    # no concatenation performed
    else:
        print('Not concatenating files.')
    
    timer_end = time.time()
    duration = timer_end - timer_start
    print('Time elapsed: ' + str(display_time(duration)) + '.')
    print('----- Done. -----')

if __name__ == '__main__':
    main(sys.argv)
