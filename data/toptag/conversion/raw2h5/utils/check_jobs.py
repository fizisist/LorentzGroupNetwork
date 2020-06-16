#  File: check_jobs.py
#  Author: Jan Offermann
#  Date: 10/10/19.

def check_jobs(number_of_runs, folder_prefix = 'run', log = 'raw2h5.log'):
    number_completed = 0
    done = False
    success = True
    string1 = 'Normal termination (return value 0)'
    string2 = 'Normal termination (return value 1)'
    for i in range(number_of_runs):
        folder_name = folder_prefix + str(i)
        logname = folder_name + '/' + log
        try: # file may not exist yet
            with open(logname, 'r') as f:
                if string1 in f.read():
                    number_completed += 1
                elif string2 in f.read():
                    number_completed += 1
                    success = False
        except: pass
    if(number_completed == number_of_runs): done = True
    
    return [done,success]
