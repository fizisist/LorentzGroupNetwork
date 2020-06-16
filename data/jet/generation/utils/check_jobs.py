#  File: display.py
#  Author: Jan Offermann
#  Date: 04/20/20.

import glob
def CheckJobs(njobs, generation_dir):
    output_files = glob.glob(generation_dir + '/**/*.root', recursive = True)
    njobs_found = len(output_files)
    done = (njobs == njobs_found)
    return done, njobs_found
