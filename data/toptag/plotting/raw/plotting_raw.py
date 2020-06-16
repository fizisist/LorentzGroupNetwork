# Plotting from raw files

import numpy as np, pandas as pd, ROOT as rt
import sys


def main(args):
    file_dir = str(sys.argv[1])
    files = [file_dir + '/' + x + '.h5' for x in ['test','train','val']]

    stores = [pd.HDFStore(x) for x in files]
    frames = [x.select('table') for x in stores]
    
    nevents = [x.shape[0] for x in frames]
    npar = 200
    px = np.zeros(npar,np.dtype('f8'))
    py = np.zeros(npar,np.dtype('f8'))
    pz = np.zeros(npar,np.dtype('f8'))
    E = np.zeros(npar,np.dtype('f8'))
    pmu = np.zeros(npar,np.dtype('f8'))

    labels = {'px':['PX_'+str(x) for x in range(npar)],'py':['PY_'+str(x) for x in range(npar)],'pz':['PZ_'+str(x) for x in range(npar)],'E':['E_'+str(x) for x in range(npar)]}
    
    pt =  np.zeros(np.sum(np.array(nevents,np.dtype('i4'))))
    eta = np.zeros(np.sum(np.array(nevents,np.dtype('i4'))))
    phi = np.zeros(np.sum(np.array(nevents,np.dtype('i4'))))

    i_global = 0
    i_global_max = np.sum(np.array(nevents,np.dtype('i4')))
    print_value = 0
    print_value_step = 10
    print('Looping through ' + str(i_global_max) + ' events.')
    for idx, N in enumerate(nevents):
        for i in range(N):
        
            if(int(i_global / i_global_max) >= print_value):
                print('\t' + str(print_value) + '%')
                print_value = print_value + print_value_step
        
            series = frames[idx].iloc[i]
            px = [series[x] for x in labels['px']]
            py = [series[x] for x in labels['py']]
            pz = [series[x] for x in labels['pz']]
            E =  [series[x] for x in labels['E']]
            
            vec = rt.TLorentzVector()
            vec.SetPxPyPzE(0.,0.,0.,0.)
            for j in range(npar):
                vec2 = rt.TLorentzVector()
                vec2.SetPxPyPzE(px[j],py[j],pz[j],E[j])
                vec = vec + vec2

            pt[i_global] = vec.Pt()
            eta[i_global] = vec.Eta()
            phi[i_global] = vec.Phi()
            i_global = i_global + 1
            
    pt_hist = rt.TH1F('pt_hist','toptag: Jet p_{T};p_{T} (GeV);Count',5000, 0., 100000.)
    eta_hist = rt.TH1F('eta_hist','toptag: Jet #eta;#eta;Count',100, -4., 4.)
    phi_hist = rt.TH1F('phi_hist','toptag: Jet #phi;#phi;Count',100, -7., 7.)
    
    for element in pt:
        pt_hist.Fill(element)
        
    for element in eta:
        eta_hist.Fill(element)

    for element in phi:
        phi_hist.Fill(element)

    file = rt.TFile('plots_py.root','RECREATE')
    
    pt_hist.Write()
    eta_hist.Write()
    phi_hist.Write()
    
    c_pt = rt.TCanvas('c_pt','c_pt',800,600)
    c_eta = rt.TCanvas('c_eta','c_eta',800,600)
    c_phi = rt.TCanvas('c_phi','c_phi',800,600)
    
    rt.gStyle.SetOptStat(0)

    c_pt.cd()
    pt_hist.Draw()
    c_pt.SetLogy()
    
    c_eta.cd()
    eta_hist.Draw()
    c_eta.SetLogy()
    
    c_phi.cd()
    phi_hist.Draw()
    c_phi.SetLogy()
    
    c_pt.Write()
    c_eta.Write()
    c_phi.Write()
    

    
    file.Close()

    
    

if __name__ == '__main__':
    main(sys.argv)

