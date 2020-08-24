# cut.py
import ROOT as rt, numpy as np
# Implement optional cuts for the kinematic plots.
top_mass = 172.76 # [GeV]
data_keys = ['Nobj','Pmu','truth_Pmu','is_signal']

def MassCut(event_info, pm = 50.):
    m = event_info['m_jet_reco']
    if(m > top_mass - pm and m < top_mass + pm): return True
    return False
    
def PassCut(event_info, type = ''):
    if(type == ''): return True
    elif(type == 'mass'): return MassCut(event_info, 50.)
    else:
        print('Cut type not understood/defined:',type)
        return True
