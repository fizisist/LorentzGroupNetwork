import numpy as np
import sys, os, glob

def grep(filename, exp): # poor man's grep
    lines = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            if exp in line:
                lines.append(line)
    return lines


# Get training times per epoch, in seconds
def GetTimes(log_file):
    # Find a line that gives the # of minibatches
    line1 = ''
    with open(log_file, 'r') as f:
        lines = f.readlines()
        for idx,line in enumerate(lines):
            if 'Starting Epoch' in line:
                line1 = lines[idx+1] # next line
                break
    minibatches = line1.split('L:')[0].split('B:')[1].strip()[:-1].split('/')[1]
    # now look for the lines where the epoch completes -> of form "minibatches/minibatches"
    lines = grep(log_file, minibatches + '/' + minibatches)
    lines = np.array([line.split('dt:')[1].split()[1] for line in lines],dtype=np.dtype('f8'))
    print('\tLogfile = ', log_file, ' \t gives ', len(lines), ' entries.')
    return lines

def main(args):
    set_folder = str(sys.argv[1]) # folder containing the runs
    run_folders = [x for x in os.listdir(set_folder) if 'run' in x]
    run_folders = [set_folder + '/' + x for x in run_folders]
    # Get the log/minilog files
    logs = []
    for run_folder in run_folders:
        logs += glob.glob(run_folder + '/*.log')
        if(glob.glob(run_folder + '/*.log') == []):
            logs += glob.glob(run_folder + '/*.minilog')
    # Get times -- one list per log file
    
    times = []
    for log in logs:
        times_tmp = GetTimes(log)
        for entry in times_tmp: times.append(entry)
    
    times = np.array(times,dtype=np.dtype('f8'))
    # Get average and standard error
    avg = np.mean(times)
    stderr = np.std(times) / np.sqrt(float(times.shape[0]))
    print('Avg   : ',avg, ' s')
    print('Stderr: ',stderr, ' s')
    return


if __name__ == '__main__':
    main(sys.argv)

    
  
