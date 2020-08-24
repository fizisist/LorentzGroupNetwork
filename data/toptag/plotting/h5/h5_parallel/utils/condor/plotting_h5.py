import numpy as np, h5py as h5, ROOT as rt, glob
import sys
from nx import GetNx
from cut import PassCut

def main(args):
    file_dir = str(sys.argv[1])
    files = glob.glob(file_dir + '/*.h5')
    files = [h5.File(x,'r') for x in files]
    
    cut_type = ''
    if(len(sys.argv) > 2): cut_type = str(sys.argv[2])
    
    # max number of particles may vary amongst files, as some have beam particles added.
    # we'll assume that all the files chosen are of the same type, i.e. this number is constant across them.
    npar = files[0]['Pmu'].shape[1] # will be 200 (no beam) or 202 (beam)
    
    labels = ['bck','sig'] # in the histogram lists, position 0 gives background, position 1 give signal (consistent with the "is_signal" flag in the data)

    # --- Histograms ---
    # Particle, detector-level (pt, eta, phi, E_t, m)
    pt_hists = [rt.TH1F('pt_hist_'+x,'toptag: Particle p_{T};p_{T} (GeV);Count',100, 0., 1000.) for x in labels] # particle pT
    eta_hists = [rt.TH1F('eta_hist_'+x,'toptag: Particle #eta;#eta;Count',100, -4., 4.) for x in labels] # particle eta
    phi_hists = [rt.TH1F('phi_hist_'+x,'toptag: Particle #phi;#phi;Count',100, -4., 4.) for x in labels] # particle phi
    et_hists = [rt.TH1F('et_hist_'+x,'toptag: Particle E_{T};E_{T} (GeV);Count',100, 0., 1000.) for x in labels] # particle et
    m_hists = [rt.TH1F('m_hist_'+x,'toptag: Particle m;m (GeV);Count',200, 0., 1000.) for x in labels] # particle mass
    
    # Number of particles per jet
    n_hists = [rt.TH1I('n_hist_'+x,'toptag: N_{particles};N ;Count',200, 0, 200) for x in labels]
    
    # Jet, detector-level (pt, eta, phi, et)
    pt_j_hists = [rt.TH1F('pt_j_hist_'+x,'toptag: Jet p_{T};p_{T} (GeV);Count',100, 0., 1000.) for x in labels] # jet pT
    eta_j_hists = [rt.TH1F('eta_j_hist_'+x,'toptag: Jet #eta;#eta;Count',100, -4., 4.) for x in labels] # jet eta
    phi_j_hists = [rt.TH1F('phi_j_hist_'+x,'toptag: Jet #phi;#phi;Count',100, -4., 4.) for x in labels] # jet phi
    et_j_hists = [rt.TH1F('et_j_hist_'+x,'toptag: Jet E_{T};E_{T} (GeV);Count',100, 0., 1000.) for x in labels] # jet et
    m_j_hists = [rt.TH1F('m_j_hist_'+x,'toptag: Jet m;m (GeV);Count',200, 0., 1000.) for x in labels] # jet mass
    
    # Nx histograms (number of jet constituents containing x% of jet energy).
    X = np.linspace(0,100,21,dtype=np.dtype('i8'))
    nx_e_hists = [{x: rt.TH1I('n'+str(x)+'_e_hist_'+y,'toptag: N^{'+str(x)+'}_{E};N^{'+str(x)+'}_{E};Count',200,0,200) for x in X} for y in labels]
    nx_et_hists = [{x: rt.TH1I('n'+str(x)+'_et_hist_'+y,'toptag: N^{'+str(x)+'}_{E_{T}};N^{'+str(x)+'}_{E_{T}};Count',200,0,200) for x in X} for y in labels]

    # 2D N histograms (number of constituents versus jet mass, pt)
    n_m_hists = [rt.TH2F('n_m_hist_'+x,'toptag: N_{particles} vs. Jet m;m (GeV);N;',100,0.,1000.,200,0.,200.) for x in labels] # n particles per jet vs. jet mass
    n_pt_hists = [rt.TH2F('n_pt_hist_'+x,'toptag: N_{particles} vs. Jet p_{T};p_{T} (GeV);N;',100,0.,1000.,200,0.,200.) for x in labels] # n particles per jet vs. jet pt

    # Signal-only histograms
    TET_hist = rt.TH1F('TET_hist','toptag: E_{T} (truth);E_{T} (truth);Count',200,0.,2000.) # truth et
    jes_hist = rt.TH1F('jes_hist','toptag: JES - 1;JES - 1;Count',100,-1.,1.) # Jet energy scale - 1
    jes_TET_hist = rt.TH2F('jes_TET_hist','toptag: JES-1 vs. E_{T} (truth);E_{T} (truth);JES;',200,0.,2000.,100,-1.,1.) # (JES-1) vs. truth et
    RET_TET_hist = rt.TH2F('RET_TET_hist','toptag: Reco E_{T} vs. Truth E_{T};E_{T} (truth);E_{T} (reco);',200,0.,2000.,200,0.,2000) # reco et vs. truth et
    TpT_hist = rt.TH1F('TpT_hist','toptag: p_{T} (truth);p_{T} (truth);Count',200,0.,2000.) # truth pT
    jps_hist = rt.TH1F('jps_hist','toptag: JpS - 1;JpS - 1;Count',100,-1.,1.) # Jet pT scale - 1
    jps_TpT_hist = rt.TH2F('jps_TpT_hist','toptag: Jp-1 vs. p_{T} (truth)S;p_{T} (truth);JpS;',200,0.,2000.,100,-1.,1.,) # (JpS-1) vs. truth pT
    RpT_TpT_hist = rt.TH2F('RpT_TpT_hist','toptag: Reco p_{T} vs. Truth p_{T};p_{T} (reco);p_{T} (truth);',200,0.,2000.,200,0.,2000.) # reco pT vs. truth pT
    n_TpT_hist = rt.TH2F('n_TpT_hist','toptag: N_{particles} vs. Truth p_{T};p_{T} (GeV);N;',200,0.,2000.,200,0.,200.) # number of particles per jet vs. jet pt

    # --- End of Histograms ---

    nevents = int(np.sum(np.array([x['Nobj'].shape[0] for x in files])))
    print('# of entries = ' + str(nevents))
    
    # Some vars we'll need for computing plots
    vec = rt.TLorentzVector()
    vec_j = rt.TLorentzVector()
    pmu = np.zeros((npar,4),np.dtype('f8'))
    
    i_global = 0
    
    # a dictionary for storing event info
    event_info = {}
    
    # Loop over files
    for file in files:
        nevents_single = file['Nobj'].shape[0]
        
        # Loop over events
        for i in range(nevents_single):
            
            event_info['N'] = file['Nobj'][i]
            event_info['sig'] = file['is_signal'][i] # 0 for background, 1 for signal
            pmu[:event_info['N'],:] = file['Pmu'][i,:event_info['N'],:]
            
            # Build the jet 4-momentum.
            vec_j.SetPxPyPzE(0.,0.,0.,0.)
            for j in range(event_info['N']):
                vec.SetPxPyPzE(pmu[j,1],pmu[j,2],pmu[j,3],pmu[j,0])
                vec_j = vec_j + vec
            event_info['pt_jet_reco'] = vec_j.Pt()
            event_info['eta_jet_reco'] = vec_j.Eta()
            event_info['phi_jet_reco'] = vec_j.Phi()
            event_info['et_jet_reco'] = vec_j.Et()
            event_info['m_jet_reco'] = vec_j.M()
            
            # Now check if event passes an optional cut. (We don't check earlier since we might need the jet var's computed above.)
            if(not PassCut(event_info, cut_type)): continue

            # Fill histograms.
            for j in range(event_info['N']):
                vec.SetPxPyPzE(pmu[j,1],pmu[j,2],pmu[j,3],pmu[j,0])
                pt_hists[event_info['sig']].Fill(vec.Pt())
                eta_hists[event_info['sig']].Fill(vec.Eta())
                phi_hists[event_info['sig']].Fill(vec.Phi())
                et_hists[event_info['sig']].Fill(vec.Et())
                m_hists[event_info['sig']].Fill(vec.M())
            
            n_hists[event_info['sig']].Fill(event_info['N'])
            pt_j_hists[event_info['sig']].Fill(event_info['pt_jet_reco'])
            eta_j_hists[event_info['sig']].Fill(event_info['eta_jet_reco'])
            phi_j_hists[event_info['sig']].Fill(event_info['phi_jet_reco'])
            et_j_hists[event_info['sig']].Fill(event_info['et_jet_reco'])
            m_j_hists[event_info['sig']].Fill(event_info['m_jet_reco'])
            
            n_m_hists[event_info['sig']].Fill(event_info['m_jet_reco'], event_info['N'])
            n_pt_hists[event_info['sig']].Fill(event_info['pt_jet_reco'], event_info['N'])
            
            # signal-only histograms
            if(event_info['sig'] == 1):
                vec.SetPxPyPzE(file['truth_Pmu'][i,1],file['truth_Pmu'][i,2],file['truth_Pmu'][i,3],file['truth_Pmu'][i,0])
                et_truth = vec.Et()
                if(et_truth != 0.): JES = event_info['et_jet_reco'] / et_truth
                else: JES = -999.;
                TET_hist.Fill(et_truth)
                jes_hist.Fill(JES - 1.)
                jes_TET_hist.Fill(et_truth,JES - 1.)
                RET_TET_hist.Fill(et_truth,event_info['et_jet_reco'])
                
                pt_truth = vec.Pt()
                if(pt_truth != 0.): JpS = event_info['pt_jet_reco'] / pt_truth
                else: JpS = -999.;
                TpT_hist.Fill(pt_truth)
                jps_hist.Fill(JpS - 1.)
                jps_TpT_hist.Fill(pt_truth,JpS - 1.)
                RpT_TpT_hist.Fill(pt_truth,event_info['pt_jet_reco'])
                n_TpT_hist.Fill(pt_truth, event_info['N'])
            
            # Make the Nx histograms, for energy and transverse energy.
            for idx, x in enumerate(X):
                Nx_E = GetNx(pmu[:event_info['N'],:], event_info['N'], x)
                Nx_ET = GetNx(pmu[:event_info['N'],:], event_info['N'], x, True)
                nx_e_hists[event_info['sig']][x].Fill(Nx_E)
                nx_et_hists[event_info['sig']][x].Fill(Nx_ET)
            
            if(i_global % 5000 == 0): print(str(i_global) + '/' + str(nevents))
            i_global = i_global + 1

    for file in files:
        file.close()
    
    # Note: We want the mean & standard error on the mean for each of the Nx histograms.
    #       However, we cannot compute these here, since we're using a parallelization scheme
    #       where this script may only be running on a subset of the data.
    #       (the samples are combined with hadd, which adds histogram bin contents)
    #       We'll compute these elsewhere, *after* concatenation.

    file = rt.TFile('plots_h5.root','RECREATE')

    for i in range(2):
        n_hists[i].Write()
        
        pt_hists[i].Write()
        eta_hists[i].Write()
        phi_hists[i].Write()
        et_hists[i].Write()
        m_hists[i].Write()

        pt_j_hists[i].Write()
        eta_j_hists[i].Write()
        phi_j_hists[i].Write()
        et_j_hists[i].Write()
        m_j_hists[i].Write()
        
        n_m_hists[i].Write()
        n_pt_hists[i].Write()

        for x in X:
            nx_e_hists[i][x].Write()
            nx_et_hists[i][x].Write()
            
    n_TpT_hist.Write()
    TET_hist.Write()
    jes_hist.Write()
    jes_TET_hist.Write()
    RET_TET_hist.Write()
    TpT_hist.Write()
    jps_hist.Write()
    jps_TpT_hist.Write()
    RpT_TpT_hist.Write()
    
    file.Close()

if __name__ == '__main__':
    main(sys.argv)

