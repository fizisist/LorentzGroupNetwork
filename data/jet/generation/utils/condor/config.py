
# config.py

# Helper functions for setting up configuration for event generation.

def GetJetParameters(path_to_file = 'jet_parameters.txt'):
    jet_vars = {} # entries are of form [value, found?, [keywords], datatype]
    jet_vars['radius'] = [0.8, False,['r','R'], 'float']
    jet_vars['power'] = [-1, False,['power','Power'], 'int']
    jet_vars['min_pt'] = [15., False, ['pT','pt','PT','Pt'], 'float']
    jet_vars['max_eta'] = [2., False, ['eta','Eta'], 'float']
    jet_vars['nconst'] = [200, False, ['const','Const'], 'int']
    
    with open(path_to_file,'r') as f:
        for line in f.readlines():
            if (not '=' in line): continue
            name = line.split('=')[0]
            value = line.split('=')[1]
            for key, val in jet_vars.items():
                for keyword in val[2]:
                    if keyword in line:
                        val[1] = True
                        if(val[3] == 'int'): val[0] = int(value)
                        else: val[0] = float(value)
    
    for key, val in jet_vars.items():
        if(val[1] == False): print('Warning: Did not find a value for ', key, ' in ', path_to_file)
    return jet_vars

def GetNEvents(path_to_file = 'config.txt'):
    with open(path_to_file,'r') as f:
        for line in f.readlines():
            if (not '=' in line): continue
            name = line.split('=')[0].strip()
            value = line.split('=')[1].replace('\n','').strip()
            if ('N' in name):
                return int(value)
    print('Warning: Did not find number of events in ', path_to_file)
    return 1

def GetNTruth(path_to_file = 'config.txt'):
    with open(path_to_file,'r') as f:
        for line in f.readlines():
            if (not '=' in line): continue
            name = line.split('=')[0].strip()
            value = line.split('=')[1].replace('\n','').strip()
            if ('n_truth' in name):
                return int(value)
    print('Warning: Did not find number of truth-level particles in ', path_to_file)
    return 10 # default

def GetNFinal(path_to_file = 'config.txt'):
    with open(path_to_file,'r') as f:
        for line in f.readlines():
            if (not '=' in line): continue
            name = line.split('=')[0].strip()
            value = line.split('=')[1].replace('\n','').strip()
            if ('n_final' in name):
                return int(value)
    print('Warning: Did not find number of final-state particles in ', path_to_file)
    return 4000 # default

def GetHadronization(path_to_file = 'config.txt'):
    with open(path_to_file,'r') as f:
        for line in f.readlines():
            if (not '=' in line): continue
            name = line.split('=')[0].strip()
            value = line.split('=')[1].replace('\n','').strip()
            if ('hadronization' in name):
                return (int(value) > 0)
    print('Warning: Did not find hadronization config in ', path_to_file)
    return True

def GetMPI(path_to_file = 'config.txt'):
    with open(path_to_file,'r') as f:
        for line in f.readlines():
            if (not '=' in line): continue
            name = line.split('=')[0].strip()
            value = line.split('=')[1].replace('\n','').strip()
            if ('mpi' in name):
                return (int(value) > 0)
    print('Warning: Did not find MPI config in ', path_to_file)
    return False

def GetRngSeed(path_to_file = 'config.txt'):
    with open(path_to_file,'r') as f:
        for line in f.readlines():
            if (not '=' in line): continue
            name = line.split('=')[0].strip()
            value = line.split('=')[1].replace('\n','').strip()
            if ('rng' in name):
                return int(value)
    print('Warning: Did not find RNG seed in ', path_to_file)
    return 1
