#  File: concat_fancy.py
#  Author: Jan Offermann
#  Date: 03/19/20.

import os, time, uuid
import subprocess as sub
from utils.concat import concat2
from utils.check_jobs import check_jobs

# concatenates a list of files in a simple, serialized manner, i.e.
# {f1, f2, ..., fn-1, fn} -> {f1, f2, ..., fn-1 + fn} -> ... -> {f1+f2+...+fn-1+fn}
def concatN(list_of_files, output_name = 'out.h5', comp_level = 5, debug = False):
    if len(list_of_files) <= 1: return
    list_of_files_dynamic = list_of_files
    tempname = ''
    tempname_old = ''
    while(len(list_of_files_dynamic) > 1):
        if(debug):
            print(len(list_of_files_dynamic), ' files left to concatenate.')
        tempname = str(uuid.uuid4().hex) + '.h5' # highly unlikely to cause namespace conflict
        concat2(list_of_files_dynamic[-1], list_of_files_dynamic[-2], tempname, True, comp_level)
        if tempname_old is not '': os.remove(tempname_old)
        list_of_files_dynamic.pop()
        list_of_files_dynamic.pop()
        list_of_files_dynamic.append(tempname)
        tempname_old = tempname
    os.rename(tempname,output_name)
    
# concatenates a list of files in a serialized, pairwise manner,
# does so recursively until all files are concatenated.
def concatN_r(input_files, output_name = 'out.h5', comp_level = 5, debug = False, delete_inputs = False, immutables = []):
    nfiles = len(input_files)
    if nfiles <= 1: return
    output_files = []
    list_of_pairs = []
    if(debug): print('inputs',input_files)
    for i in range(int(nfiles/2)):
        list_of_pairs.append((input_files[2*i], input_files[2*i + 1]))
        if(debug): print('pairs[' + str(i) + '] = ', list_of_pairs[-1])
    
    # If nfiles is odd, there is one unpaired file. Put this in the front of output_files
    if(int(nfiles / 2) * 2 != nfiles):
        output_files.append(input_files[-1])
    
    for pair in list_of_pairs:
        tempname = str(uuid.uuid4().hex) + '.h5'
        concat2(pair[0],pair[1], tempname, True, comp_level)
        output_files.append(tempname)
    
    # Make list of non-deletable inputs -- these correspond with inputs at the top of the recursion
    if(not delete_inputs):
        for file in input_files:
            immutables.append(file)
    
    # Delete input files
    if(delete_inputs):
        for i in range(nfiles):
            if(int(nfiles / 2) * 2 != nfiles and i == nfiles-1): break # don't delete last file if nfiles is odd
            if(input_files[i] not in immutables): os.remove(input_files[i])

    if(debug): print('outputs', output_files)
    
    nfiles = len(output_files)
    if nfiles > 1: concatN_r(output_files, output_name, comp_level, debug, True, immutables)
    if nfiles == 1: os.rename(output_files[0], output_name)
    return
    
# Concat using Condor -- submits concat2 jobs to Condor to concat files pairwise.
def concatN_condor(input_files, output_name = 'out.h5', comp_level = 9, debug = False, concat_template = 'template/concat_template.sub', concat_sub = 'concat.sub'):
    
    # For Python2 compatibility, let's get DEVNULL for suppressing some outputs of subprocess.check_call()
    FNULL = open(os.devnull, 'w')
    
    nfiles = len(input_files)
    output_files = []
    list_of_pairs = []
    if(debug): print('inputs',input_files)
    for i in range(int(nfiles/2)):
        list_of_pairs.append((input_files[2*i], input_files[2*i + 1]))
        if(debug): print('pairs[' + str(i) + '] = ', list_of_pairs[-1])
    
    njobs = len(list_of_pairs)
    # If nfiles is odd, there is one unpaired file. Put this in the front of output_files
    if(int(nfiles / 2) * 2 != nfiles):
        output_files.append(input_files[-1])
        if(debug): print('nfiles is odd, not concatenating ' + input_files[-1])
    
    # make the condor submit file, based on the template
    try: sub.check_call(['rm',concat_sub],stdout=FNULL, stderr=sub.STDOUT)
    except: pass
    filenames = ['file1.h5','file2.h5']
    text = ''
    with open(concat_template,'r') as f:
        text = f.readlines()
    with open(concat_sub, 'w') as f:
        for line in text:
            line = line.replace('$FILE1', filenames[0])
            line = line.replace('$FILE2', filenames[1])
            line = line.replace('$OUTPUT', output_name)
            line = line.replace('$COMP_LEVEL', str(comp_level))
            line = line.replace('$NJOBS', str(njobs))
            f.write(line)
    
    # make a directory for each condor job, copy relevant pair of input files into that directory
    for idx, pair in enumerate(list_of_pairs):
        dirname = 'concat_job' + str(idx) # directory name based on assumptions in concat submission template
        dirname_temp = str(uuid.uuid4().hex)
        f1_t = dirname_temp + '/' + filenames[0]
        f2_t = dirname_temp + '/' + filenames[1]
        f1 = dirname + '/' + filenames[0]
        f2 = dirname + '/' + filenames[1]
        
        # copy input files to temp directory -- in some cases their original locations will be deleted!
        sub.check_call(['mkdir',dirname_temp])
        sub.check_call(['cp',pair[0],f1_t])
        sub.check_call(['cp',pair[1],f2_t])
        
        # overwrite existing Condor job directories -- this may delete original input files, but they have been copied above
        try:
            sub.check_call(['rm','-r',dirname],stdout=FNULL, stderr=sub.STDOUT)
            if(debug): print('Overwriting directory: ' + dirname)
        except: pass
        sub.check_call(['mkdir', dirname])
        
        # move input files to Condor job directories, delete temp directory
        if(debug):
            print('Moving ' + f1_t + ' -> ' + f1)
            print('Moving ' + f2_t + ' -> ' + f2)
        sub.check_call(['mv',f1_t,f1])
        sub.check_call(['mv',f2_t,f2])
        sub.check_call(['rm','-r',dirname_temp])
        output_files.append(dirname + '/' + output_name)
        
    # now submit the concatenation job to Condor
    sub.check_call(['condor_submit', concat_sub])
    if(debug): print('Submitting '+str(njobs)+' concat jobs to Condor.')
    return output_files # returns list of output files -- these do not necessarily exist yet!
  
# Use concatN_condor recursively to concatenate all files TODO: Consider using kwargs for so many arguments
def concatN_condor_r(input_files, output_name = 'out.h5', nfiles_final = 1, comp_level = 9, debug = False, concat_template = 'template/concat_template.sub', concat_sub = 'concat.sub'):
    
    nfiles = len(input_files)
    if nfiles <= nfiles_final:
        for idx, file in enumerate(input_files):
            output_name_temp = output_name.replace('.h5','_'+str(idx)+'.h5')
            sub.check_call(['mv',file,output_name_temp]) # move the last file out of the concat job directory
        try: sub.check_call('rm -r concat_job*',shell=True) # delete concat job directories
        except: pass
        return
        
    njobs = int(nfiles / 2)
    if(debug): print('njobs = ' + str(njobs))
    # concatenate pairwise, get list of output files (will include leftover file if nfiles is odd)
    output_files = concatN_condor(input_files, output_name, comp_level, debug, concat_template, concat_sub)
    
    # Check for job completion
    status = False
    success = True
    max_wait_time = 21600 # seconds ( = 6 hours)
    sleep_time = 60 # seconds ( = 1 minute)
    start = time.time()
    while True:
        time.sleep(sleep_time) # sleep
        [status,success] = check_jobs(njobs, 'concat_job','concat.log') # check log files of jobs
        if(status == True): break # if jobs are completed, exit loop
        now = time.time()
        if(now - now > max_wait_time): break # exit loop if too much time passed
    if (status == False):
        print('Uh oh, some concat jobs didn\'t complete on-time.')
        return
    if (not success):
        print('Uh oh, some concat jobs failed.')
        return

    # Jobs have completed, now run again on the results
    concatN_condor_r(output_files,output_name,nfiles_final,comp_level,debug,concat_template,concat_sub)
