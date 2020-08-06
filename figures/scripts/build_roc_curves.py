# This script is like a "manager", it actually calls
# a script in ../ROC_curves/ to build the ROC curves.

import sys, os, glob
import subprocess as sub
import warnings
warnings.filterwarnings("ignore")

def main(args):
    
    set_dir = sys.argv[1] # folder containing run folders -- each must contain a set of .csv files corresponding with ROC curve info
    
    test_file = sys.argv[2] # raw dataset test file
    use_test_file = 1
    if(test_file == 'NONE'): use_test_file = 0
    
    doTest = int(sys.argv[3])
    if(doTest == 0): doTest = False # if false, use latest validation ROC data. Otherwise use the test data.
    else: doTest = True
    
    if(len(sys.argv) > 4):
        doAllValid = int(sys.argv[4])
        if(doAllValid == 0): doAllValid = False
        else: doAllValid = True
    else: doAllValid = False # whether to make ROC curves for just the latest validation or all epochs
    
    git_fig_dir = os.path.dirname(os.path.realpath(__file__)) + '/..'
    roc_script = git_fig_dir + '/ROC_curves/' + 'buildTopTagROC.py'
    datadir_ref = git_fig_dir + '/ROC_curves/TopTagReference/'
    roc_name = 'ROC_'
    
    if(doTest): roc_name = roc_name + 'test'
    else: roc_name = roc_name + 'valid_epoch_'

    roc_files = []
        
    run_dirs = [x for x in glob.glob(set_dir + '/run*') if ('.' not in x.split('/')[-1])] # remove files from list
    # for each run, we want the last ROC csv file
    for run in run_dirs:
        if(not doTest):
            csv_files = [x for x in glob.glob(run + '/' + '*valid_ROC.csv')]
            if(csv_files == []): continue
            numbers = [int(x.split('epoch')[-1].split('.')[0]) for x in csv_files]
            if(not doAllValid):
                max_number = max(numbers)
                roc_files.append((csv_files[numbers.index(max_number)],max_number))
            else:
                for idx, val in enumerate(csv_files):
                    roc_files.append((val,numbers[idx]))
        else:
            csv_files = [x for x in glob.glob(run + '/' + '*test_ROC.csv')]
            for x in csv_files: roc_files.append((x,-1))
    
#    print('ROC files:')
#    for x in roc_files:
#        print('\t'+x[0])

    for entry in roc_files:
        roc_file = entry[0]
        run_number = entry[1]
        roc_dir = roc_file.replace(roc_file.split('/')[-1],'')[0:-1]
        roc_dir = roc_dir + '/ROCcomparisons'
        try: os.makedirs(roc_dir)
        except: pass
        
        roc_name_tmp = roc_name
        if(run_number != -1):
            addition = str(run_number)
            if(run_number < 10): addition = '0' + addition
            roc_name_tmp = roc_name_tmp + addition
        
        # Now, we actually only want to generate ROC curves that we haven't *already* generated.
        existing_ROC_curve = glob.glob(roc_dir + '/' + roc_name_tmp + '*')
        if(len(existing_ROC_curve) > 0): continue
        print('Creating ROC curve for ' + roc_file + '\t at ' + roc_dir + '/' + roc_name_tmp)
        sub.check_call(['python',roc_script,roc_file,datadir_ref,roc_dir,roc_name_tmp,str(use_test_file),test_file])

            
if __name__ == '__main__':
    main(sys.argv)

        
    
