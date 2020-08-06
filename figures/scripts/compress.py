# For compressing the csv (ROC curve) and pt (weight) files, which we don't need
# for perf_plots.

import numpy as np, subprocess as sub
import sys, os, glob, uuid, re

# see https://stackoverflow.com/a/480227
def f7(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

# Create smaller log files for perf_plot, to give a speedup when regenerating plots.
# Since the parent log file might be updated with more info (especially if it is used
# before all epochs are finished), the minilog will store the modification timestamp
# of the log, so that we can recreate it if there have been changes to the log file.
#
# As of the introduction of the non-verbose log file option ('v=0'), the mininlogs
# are no longer necessary, but we will keep the functionality to deal with the possibility
# of training sessions run without this argument.
def MakeMiniLog(filename):
    minilog = filename.replace('.log','.minilog')
    # Strings to look for, we will also save some additional lines for full functionality:
    comp_string = 'Complete!'
    lines = 0
    with open(filename,'r') as f:
        lines = f.readlines()
    minilog_lines = []
    
    # Let's save the modification timestamp of the logfile (we append this at the end)
    log_timestamp = str(os.path.getmtime(filename))
    log_timestamp = 'LOG TIMESTAMP: ' + log_timestamp
    
    # Let's save the first 40 lines to start. This is arbitrary but they will include some useful info at the start.
    N = 40
    for i in range(N):
        minilog_lines.append(lines[i].strip('\n')) # remove the \n at the end
    # Now we'll look for the completion of epochs, and save them, plus the surrounding 10 lines for each.
    # This method will pick up some duplicates, which we'll clear out at the end for tidiness.
    M = 10
    for idx,line in enumerate(lines[N:]):
        if(comp_string in line):
            min_idx = int(np.maximum(idx-M,0))
            max_idx = int(np.minimum(idx+M,len(lines[N:])-1))
            for i in range(min_idx, max_idx+1, 1):
                minilog_lines.append(lines[N:][i].strip('\n'))
    # Get rid of duplicate entries
#    minilog_lines = list(set(minilog_lines)) # This may scramble order, so that it is not very human-readable
    minilog_lines = f7(minilog_lines)
    minilog_lines.append(log_timestamp)
    # Write to minilog file
    with open(minilog,'w') as f:
        for line in minilog_lines: print(line,file=f)
    return

def CheckTimestamp(log, minilog):
    minilog_timestamp = 0
    with open(minilog,'r') as f:
        minilog_timestamp = f.readlines()[-1]
    minilog_timestamp = float(minilog_timestamp.split(':')[-1].strip())
    log_timestamp = os.path.getmtime(log)
    if(log_timestamp == minilog_timestamp): return True
    return False

# simplify filenames for readability
def SimplifyPath(path):
    path = re.sub(r'/[^/]*/\.\./', r'/', path)
    if path is re.sub(r'/[^/]*/\.\./', r'/', path):
        return path
    else:
        return SimplifyPath(path)

def compress(run_folders):
    dirnames = ['csv','pt']
    for run_folder in run_folders:
        bz2s = glob.glob(run_folder + '/*.bz2')
        if(len(bz2s) > 0): continue # assume this folder has already been compressed
        for dirname in dirnames:
            try: os.makedirs(run_folder + '/' + dirname)
            except: pass
            sub.check_call('mv ' + run_folder + '/*.' + dirname + ' ' + run_folder + '/' + dirname + '/',shell=True)
            sub.check_call('tar -jcvf ' + dirname + '.tar.bz2 ' + dirname,shell=True, cwd=run_folder,stdout = sub.DEVNULL)
            sub.check_call(['rm','-r',run_folder + '/' + dirname])
    return
      
def decompress(run_folders):
    dirnames = ['csv','pt']
    for run_folder in run_folders:
        print(run_folder)
        for dirname in dirnames:
            random_name = str(uuid.uuid4())
            sub.check_call(['mkdir',random_name])
            print(['tar','-jxvf', run_folder + '/' + dirname + '.tar.bz2','-C',random_name])
            sub.check_call('tar -jxvf '+ run_folder + '/' + dirname + '.tar.bz2 ' + '-C ' + random_name,shell=True,stdout = sub.DEVNULL)
            sub.check_call('mv ' + random_name + '/' + dirname + '/*.' + dirname + ' ' + run_folder + '/', shell=True)
            sub.check_call(['rm','-r',random_name])
            sub.check_call(['rm',run_folder + '/' + dirname + '.tar.bz2'])

def main(args):
    path_to_this_macro_dir = os.path.dirname(os.path.realpath(__file__))
    set_folder = str(sys.argv[1]) # folder containing the runs
    compression = int(sys.argv[2]) # 0 = compress, 1 = decompress, 2 = make minilogs & delete logs
    run_folders = [x for x in os.listdir(set_folder) if 'run' in x]
    run_folders = [set_folder + '/' + x for x in run_folders]
    run_folders = [SimplifyPath(run_folder) for run_folder in run_folders] # tar doesn't like ".." in paths

    if(compression == 0): compress(run_folders)
    if(compression == 1): decompress(run_folders)
    if(compression == 2):
        for run_folder in run_folders:
            try:
                log = glob.glob(run_folder + '/*.log')[0]
                MakeMiniLog(log)
                sub.check_call(['rm', log])
            except: pass
    return
    
if __name__ == '__main__':
    main(sys.argv)

    
  
