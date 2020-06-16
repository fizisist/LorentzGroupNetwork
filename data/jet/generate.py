#  File: generate.py
#  Author: Jan Offermann
#  Date: 04/17/20.
#  Goal: Generate training data samples for the Lorentz Group Network.

import sys, os, time, glob
import subprocess as sub
from generation.utils.simplify import SimplifyPath
from generation.utils.display import DisplayTime
from generation.utils.check_jobs import CheckJobs
import ROOT as rt #TODO: Can remove once TTree::fUserInfo issue is fixed

def run_condor(submit_file, generate_path, max_wait):
    # Submit the jobs to condor.
    sub.check_call(['condor_submit',submit_file],cwd = generation_path)
    print('Jobs submitted to Condor. Good luck!')
    check_interval = 15 # check every 15 seconds
    condor_done = False
    njobs_done = 0
    while(time.time() - condor_start < max_wait):
        time.sleep(check_interval)
        njobs_done_prev = njobs_done
        condor_done, njobs_done = CheckJobs(njobs, generation_path)
        if(condor_done): break
        if(njobs_done > njobs_done_prev):
            print('Update: Finished ', njobs_done, '/', njobs, ' jobs.')
    return condor_done


def run_local(submit_file, job_dirs, parallel = False):
    
    # Determine which files to copy to the job directories
    copy_files = 0
    with open(submit_file,'r') as f:
        text = f.readlines()
        for line in text:
            if('transfer_input_files' in line):
                copy_files = (line.split('=')[-1])
                break
    if(copy_files == 0):
        print('Error: Did not find copy files.')
        return
    copy_files = copy_files.strip().split(',') # files are relative to the job dirs
    copy_files = [x for x in copy_files if ('/' in x)] # remove files from list that are already in the job dirs
    
    # Copy files to the job dirs, to recreate condor worker environments.
    for job_dir in job_dirs:
        cp_command = ['cp',0,job_dir+'/']
        for file in copy_files:
            cp_command[1] = job_dir + '/' + file
            sub.check_call(cp_command)
            
    # Run the generation job locally in each folder.
    generate_command = ['python','generator.py']
    for job_dir in job_dirs:
        if(parallel): sub.Popen(generate_command,cwd=job_dir) # TODO: parallelization is experimental, need way to check completion
        else: sub.check_call(generate_command,cwd=job_dir)
    return
    

    

def main(args):
    if(len(sys.argv) > 1):
        delete_runs = (int(sys.argv[1]) > 0)
    else: delete_runs = False
    
    do_condor = False
    
    # Path to this script's directory
    path_to_this = os.path.dirname(os.path.realpath(__file__))
    
    # Paths to some necessary directories
    config_path = SimplifyPath(path_to_this + '/config')
    generation_path = SimplifyPath(path_to_this + '/generation')
    conversion_path = SimplifyPath(path_to_this + '/root2h5')
    
    # Get the template configuration files.
    config_files = {
    'Top_Wqq' : config_path + '/samples/Top_Wqq',
    'Top' : config_path + '/samples/Top',
    'QCD' : config_path + '/samples/QCD'
    }
    
    config_files = {key : glob.glob(config_files[key] + '/*.[Tt][Xx][Tt]')[0] for key in config_files.keys()}
    
    # Determine what processes to run, and make the appropriate
    # job directories to be sent to condor. We will use 1 job
    # per configuration. This could be increased but would require
    # careful adjustment of the RNG seeds, otherwise we would get
    # duplicate events.
    # We will gather other pieces of info such as jet clustering params.
    general_config = config_path + '/config.txt'
    configs = {}
    configs['processes'] = []
    configs['pt_cutoffs'] = []
    with open(general_config, 'r') as f:
        config_contents = f.readlines()
        for line in config_contents:
            line = line.split('#')[0]
            if('process' in line and '=' in line):
                configs['processes'] += line.split('=')[-1].split(',')
            elif('cutoff' in line and '=' in line):
                configs['pt_cutoffs'] += line.split('=')[-1].split(',')
            elif('Nevent' in line and '=' in line):
                configs['N'] = line.split('=')[-1].strip(' ')
            elif('power' in line and '=' in line):
                configs['power'] = line.split('=')[-1].strip(' ')
            elif('rad' in line and '=' in line):
                configs['R'] = line.split('=')[-1].strip(' ')
            elif('pT_min' in line and '=' in line):
                configs['pt_min'] = line.split('=')[-1].strip(' ')
            elif('eta' in line and '=' in line):
                configs['eta'] = line.split('=')[-1].strip(' ')
            elif('const' in line and '=' in line):
                configs['nconst'] = line.split('=')[-1].strip(' ')
            elif('hadronization' in line and '=' in line):
                configs['hadron'] = line.split('=')[-1].strip(' ')
            elif('mpi' in line and '=' in line):
                configs['mpi'] = line.split('=')[-1].strip(' ')
            elif('rng' in line and '=' in line):
                configs['rng'] = line.split('=')[-1].strip(' ')
            elif('n_truth' in line and '=' in line):
                configs['n_truth'] = line.split('=')[-1].strip(' ')
            elif('n_final' in line and '=' in line):
                configs['n_final'] = line.split('=')[-1].strip(' ')

    # Clean up the lists.
    configs['processes'] = [x.replace(' ','').replace('\n','') for x in configs['processes']]
    configs['pt_cutoffs'] = [x.replace(' ','').replace('\n','') for x in configs['pt_cutoffs']]

    # Make the job directories.
    njobs = len(configs['processes']) * (len(configs['pt_cutoffs']) - 1)
    job_dirs = [generation_path + '/job' + str(x) for x in range(njobs)]
    for x in job_dirs:
        try: os.makedirs(x)
        except: pass
    
    job_iterator = 0
    
    # Copy Pythia8 config's to the job folders.
    for process in configs['processes']:
        if(process not in config_files.keys()): continue
        template = config_files[process]
        
        for i in range(len(configs['pt_cutoffs'])-1):
            min_pt = str(configs['pt_cutoffs'][i])
            max_pt = str(configs['pt_cutoffs'][i+1])
            
            pythia_config = 'pythia_config.txt'
            pythia_config = generation_path + '/job' + str(job_iterator) + '/' + pythia_config
            
            with open(template,'r') as f:
                text = f.readlines()
            with open(pythia_config,'w') as f:
                for line in text:
                    line=line.replace('$MINIMUM_PT',min_pt)
                    line=line.replace('$MAXIMUM_PT',max_pt)
                    if('inf' not in line): f.write(line)
            job_iterator += 1
    
    # Copy the jet clustering parameters to the job folders.
    for x in job_dirs:
        jet_config = x + '/jet_parameters.txt'
        with open(jet_config,'w') as f:
            f.write('power = ' + str(configs['power']))
            f.write('R = ' + str(configs['R']))
            f.write('pT_min = ' + str(configs['pt_min']))
            f.write('eta_max = ' + str(configs['eta']))
            f.write('n_constituents = ' + str(configs['nconst']))
       
    # Copy the general config parameters to the job folders.
    for x in job_dirs:
        run_config = x + '/config.txt'
        with open(run_config,'w') as f:
            f.write('N = ' + str(configs['N']))
            f.write('hadronization = ' + str(configs['hadron']))
            f.write('mpi = ' + str(configs['mpi']))
            f.write('rng = ' + str(configs['rng']))
            f.write('n_truth = ' + str(configs['n_truth']))
            f.write('n_final = ' + str(configs['n_final']))


    # Set up the condor submit file, using the template.
    submit_file = generation_path + '/generate.sub'

    with open(generation_path + '/template/generate_template.sub', 'r') as f:
        text = f.readlines()
    with open(submit_file, 'w') as f:
        for line in text:
            line = line.replace('$NJOBS',str(njobs))
            f.write(line)
    
    
    # Ship jobs to condor, or run locally.
    
    # If do_condor = True (default), we submit the condor jobs.
    if(do_condor):
        max_wait = 24 * 60 * 60 # max wait is 24 hours
        condor_start = time.time()
        condor_done = run_condor(submit_file,generate_path, max_wait)
        condor_stop = time.time()
        if(not condor_done):
            print('Error: Failed to finish jobs after max wait time: ', DisplayTime(max_wait), '.')
            return
        print('Completed all jobs. Conversion time: ', DisplayTime(condor_stop - condor_start))

    # If do_condor = False, we will run a bunch of local jobs (serialized).
    else: run_local(submit_file,job_dirs)
        
        
    # Perform reduction on all the ROOT output files.
    # Filenames of the generation output will be "output.root".
    # Filenames of the reduction output will be "output_reduced.root"
    output_files = [x + '/out.root' for x in job_dirs]
    reduction_command = ['python',conversion_path + '/reduction.py',0]
    for output_file in output_files:
        reduction_command[-1] = output_file
        sub.check_call(reduction_command)

    # Make a final output folder. The concatenated, reduced ROOT file will go here,
    # together with the HDF5 file (that will be our final output).
    output_dir = path_to_this + '/output'
    try: os.makedirs(output_dir)
    except: pass
    
    # Make a subdir in the final output folder. It will be named "runX",
    # where X will be an integer to identify this particular run.
    
    output_subdir = output_dir + '/run'
    X = 0
    while(True):
        if(not os.path.exists(output_subdir + str(X))):
            output_subdir = output_subdir + str(X)
            break
        X += 1
    os.makedirs(output_subdir)
    
    # Concatenate all the reduced ROOT output files.
    reduced_output_files = [x + '/out_reduced.root' for x in job_dirs]
    reduction_output_name = output_subdir + '/events.root'
    hadd_command = ['hadd', reduction_output_name]
    [hadd_command.append(x) for x in reduced_output_files]
    sub.check_call(hadd_command)
    
    # Pick up the TTree::fUserInfo (TODO: should shift this to reduction.py, once hadd is fixed. See: https://sft.its.cern.ch/jira/browse/ROOT-10763)
    event_single_tfile = rt.TFile(output_files[0],'READ')
    event_single_ttree = event_single_tfile.Get('events')
    event_tfile = rt.TFile(reduction_output_name,'UPDATE')
    event_ttree = event_tfile.Get('events_reduced')
    for param in event_single_ttree.GetUserInfo():
            event_ttree.GetUserInfo().Add(param)
    event_ttree.Write('',rt.TObject.kOverwrite) # should be automatically cd'd to event_tfile
    event_tfile.Close()
    event_single_tfile.Close()
    
    # Split the reduced ROOT file into train/test/validation samples.
    ttv_command = ['python',conversion_path +'/ttv_split.py',reduction_output_name]
    sub.check_call(ttv_command)
    reduction_output_names = [reduction_output_name.replace('.root','_'+x+'.root') for x in ['train','test','valid']]

    # ROOT -> HDF5 conversion
    conversion_command = ['python', conversion_path + '/root2h5.py', 0]
    for name in reduction_output_names:
        conversion_command[-1] = name
        sub.check_call(conversion_command)
        
    # -- Cleanup Pt 1: pycache ---
    
    # Delete any pesky __pycache__ folders left over,
    # these can be regenerated on the next run and
    # we want to avoid clutter.
    pycaches = glob.glob(path_to_this + '/**/__pycache__', recursive = True)
    [sub.check_call(['rm','-r',pycache]) for pycache in pycaches]
    
    # Delete any files created from compiling TPythia8,
    # if generation was run locally.
    comp_files =  glob.glob(path_to_this + '/**/*.d', recursive = True)
    comp_files += glob.glob(path_to_this + '/**/*.pcm', recursive = True)
    [sub.check_call(['rm','-r',comp_file]) for comp_file in comp_files]
    
    # -- Cleanup Pt2: ROOT ttv files
    
    # Clean up the events_[train,test,valid].root files.
    # We will keep the full reduced ROOT file for validation purposes.
    [sub.check_call(['rm',x]) for x in reduction_output_names]
    
    # -- Cleanup Pt 3: job folders (optional) ---
    
    # This will remove the job folders entirely,
    # deleting the raw event files (that contain the full event info)
    # plus the uncombined, reduced ROOT files (which are redundant).
    if(delete_runs):
        for x in job_dirs:
            sub.check_call(['rm','-r',x])
        sub.check_call(['rm',submit_file])
    return
    
if __name__ == '__main__':
    main(sys.argv)
