#!/usr/bin/env python

import os, sys

job_dir = '../.tmp'

template_start="""#!/bin/bash
#SBATCH --job-name=LGN
#SBATCH --chdir=/NBodyJetNets/NetworkDesign/
#SBATCH --time=36:00:00
#SBATCH --partition=river-gpu
#SBATCH --account=pi-aachien
#SBATCH --gres=gpu:1
#SBATCH --export=ALL
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bogatsky@uchicago.edu
#SBATCH --mem=8gb 
#SBATCH --output=slurm/out/slurm-%j.out

echo GPU ID: $CUDA_VISIBLE_DEVICES
echo "Beginning GM4 job."

module load Anaconda3/2018.12
source activate pt

"""

template_end="""

echo Cleaning up file: ${1}

rm ${1}

echo "Job complete!"
"""

jobs = {}

verbose = False

with open(sys.argv[1], 'r') as f:
    for line in f:
        if line.startswith('python3'):
            prefix='--prefix'
            jobname = [word[9:] for word in line.split() if word.startswith(prefix)]
            assert(len(jobname) == 1),'{}'.format(jobname)
            jobname = jobname[0]
            jobs[jobname]=line.rstrip()

for name, job in jobs.items():
    job_file = job_dir + '/' + name + '.job'
    print('Creating job {} in file {}'.format(name, job_file))

    with open(job_file, 'w') as f:
        f.writelines(template_start)
        f.writelines('\n')
        f.writelines(job)
        f.writelines('\n')
        f.writelines('\n')
        f.writelines(template_end)

    if verbose:
        print(job)

    os.chmod(job_file, 0o775)

    print('Submitting job.', end=' ', flush=True)
    os.system("sbatch {}".format(job_file))
    print('Done!')

print('All jobs done!')

