#  present.py
# Created by Jan Offermann on 01/04/20.
# Goal: Draw the histograms created from the toptag dataset.
#       (Pythonized & updated version of "present.C")

import ROOT as rt, numpy as np, sys

# Helper function for making TPaveText
def SetupPave(x1, y1, x2, y2):
    pave = rt.TPaveText(x1,y1,x2,y2, 'NDC')
    pave.SetTextFont(42)
    pave.SetTextAngle(0)
    pave.SetTextColor(rt.kBlack)
    pave.SetTextSize(0.04)
    pave.SetTextAlign(12)
    pave.SetFillStyle(0)
    pave.SetBorderSize(0)
    return pave

def SetupLegend(x1, y1, x2, y2):
    leg = rt.TLegend(x1,y1,x2,y2)
    leg.SetFillColor(0)
    leg.SetFillStyle(0) # make it transparent!
    leg.SetBorderSize(0)
    leg.SetTextFont(42)
    leg.SetTextAngle(0)
    leg.SetTextColor(rt.kBlack)
    leg.SetTextSize(0.03)
    leg.SetTextAlign(12)
    return leg

def PlotAdjust(histogram, ranges, title = '', scientific = False):
    label_size = 0.05
    title_size = 0.05
    title_offset = [1.,1.25,1.]
    line_thickness_multiplier = 2
    naxes = len(ranges) # 2 for 1D histogram, 3 for 2D histogram (# of axes)
    if(naxes < 2):
        print('Warning: Histogram has ndim = ', naxes - 1)
    axes = [histogram.GetXaxis(),histogram.GetYaxis()]
    if(naxes == 3): axes.append(histogram.GetZaxis())
    [x.SetLabelSize(label_size) for x in axes]
    [x.SetTitleSize(title_size) for x in axes]
    for i in range(naxes):
        axes[i].SetTitleOffset(title_offset[i])
        axes[i].SetRangeUser(ranges[i][0],ranges[i][1])
    if(not title == ''): histogram.SetTitle(title)
    if(scientific):
        for i in range(1,naxes):
            axes[i].SetMaxDigits(3)
    return

def PlotDisplay(c, plots, leg, pave, log = True, d2 = False, errors = False):
    nplots = len(plots)
    c.cd()
    option = ''
    if(d2):
        option = 'COLZ'
        c.SetRightMargin(0.125) # leave space for colorbar labels (sufficient for log scale)
    if(errors): option = 'HIST E1'
    plots[-1].Draw(option)
    for i in range(nplots-2,-1,-1):
        if(option == ''): option = 'SAME'
        else: option += ' SAME' # TODO: Space seems to matter!
        plots[i].Draw(option)
    if(leg != 0): leg.Draw()
    pave.Draw()
    if(log):
        if(d2): c.SetLogz()
        else: c.SetLogy()
    c.Draw()
    c.Update()
    return

def CanvasSave(c, prefix, f, suffixes = ['.eps','.png','.pdf']):
    for suffix in suffixes:
        c.SaveAs(prefix + suffix)
    f.cd()
    c.Write('',rt.TObject.kOverwrite)
    return

# Make plot of avg. Nx as a function of x
def AvgNx(f, X, suffixes, transverse = False):
    if(transverse): nx_hists = [{x:f.Get('n'+str(x)+'_e_hist_'+y) for x in X} for y in suffixes]
    else: nx_hists = [{x:f.Get('n'+str(x)+'_e_hist_'+y) for x in X} for y in suffixes]
    mean = np.full((2,len(X)),-1.,dtype=np.dtype('f8'))
    stderr = np.full((2,len(X)),-1.,dtype=np.dtype('f8'))
    for idx, x in enumerate(X):
        for y in range(2):
            mean[y,idx] = nx_hists[y][x].GetMean()
            stderr[y,idx] = nx_hists[y][x].GetMeanError()
    if(transverse): nx_avg_hists = [rt.TH1F('nx_et_hist_'+y,'toptag: #bar{N^{x}_{E_{T}}};x (%);#bar{N^{x}_{E_{T}}}',20, 0, 100) for y in suffixes]
    else: nx_avg_hists = [rt.TH1F('nx_e_hist_'+y,'toptag: #bar{N^{x}_{E}};x (%);#bar{N^{x}_{E}}',20, 0, 100) for y in suffixes]
    for y in range(2):
        for idx, x in enumerate(X):
            nx_avg_hists[y].SetBinContent(idx, mean[y,idx])
            nx_avg_hists[y].SetBinError(idx, stderr[y,idx])
    return nx_avg_hists

def main(args):
    if(len(sys.argv) > 1 ): filename = str(sys.argv[1])
    else: filename = 'toptag.root'
    
    # QoL options
    rt.gROOT.SetBatch(True)
    rt.gROOT.ProcessLine('gErrorIgnoreLevel = kInfo+1;')
    
    # Setup ATLAS Style
    rt.gROOT.SetStyle('ATLAS') # works since ROOT version 6.13
    rt.gROOT.ForceStyle()
    # Some style adjustments
    rt.gStyle.SetHistLineWidth(1) # no bold lines, these are too thick
    rt.gStyle.SetMarkerStyle(rt.kDot) # effectively erase markers, but keeps error bar caps (?)
    rt.gStyle.SetMarkerSize(0.75)
    rt.gStyle.SetEndErrorSize(2.)
    
    f = rt.TFile(filename,'READ')
    suffixes = ['bck','sig']
    
    # Jet histograms
    pt_j = [f.Get('pt_j_hist_'+x) for x in suffixes]
    pt_j[0].SetLineColor(rt.kRed)
    pt_j[1].SetLineColor(rt.kBlue)
    
    eta_j = [f.Get('eta_j_hist_'+x) for x in suffixes]
    eta_j[0].SetLineColor(rt.kRed)
    eta_j[1].SetLineColor(rt.kBlue)
    
    phi_j = [f.Get('phi_j_hist_'+x) for x in suffixes]
    phi_j[0].SetLineColor(rt.kRed)
    phi_j[1].SetLineColor(rt.kBlue)
    
    et_j = [f.Get('et_j_hist_'+x) for x in suffixes]
    et_j[0].SetLineColor(rt.kRed)
    et_j[1].SetLineColor(rt.kBlue)
    
    m_j = [f.Get('m_j_hist_'+x) for x in suffixes]
    m_j[0].SetLineColor(rt.kRed)
    m_j[1].SetLineColor(rt.kBlue)
    
    nobj = [f.Get('n_hist_'+x) for x in suffixes]
    nobj[0].SetLineColor(rt.kRed)
    nobj[1].SetLineColor(rt.kBlue)
    
    X = np.linspace(0,100,21,dtype=np.dtype('i8')) #TODO: This must match the array in plotting_h5.py
    nx_e = [{x: f.Get('n'+str(x)+'_e_hist_'+y) for x in X} for y in suffixes]
    [nx_e[0][x].SetLineColor(rt.kRed) for x in X]
    [nx_e[1][x].SetLineColor(rt.kBlue) for x in X]
    nx_et = [{x: f.Get('n'+str(x)+'_et_hist_'+y) for x in X} for y in suffixes]
    [nx_et[0][x].SetLineColor(rt.kRed) for x in X]
    [nx_et[1][x].SetLineColor(rt.kBlue) for x in X]

    # Make the average Nx distributions, using the Nx distributions above
    nx_e_avg = AvgNx(f,X, suffixes, False)
    nx_e_avg[0].SetLineColor(rt.kRed)
    nx_e_avg[0].SetMarkerColor(rt.kRed)
    nx_e_avg[1].SetLineColor(rt.kBlue)
    nx_e_avg[1].SetMarkerColor(rt.kBlue)
    
    nx_et_avg = AvgNx(f,X, suffixes, True)
    nx_et_avg[0].SetLineColor(rt.kRed)
    nx_et_avg[0].SetMarkerColor(rt.kRed)
    nx_et_avg[1].SetLineColor(rt.kBlue)
    nx_et_avg[1].SetMarkerColor(rt.kBlue)
    # --------------------
    
    # N vs m_j histogram
    n_m = f.Get('n_m_hist')
    
    # Jet energy scale & other signal-only histograms
    jes = f.Get('jes_hist')
    jes.SetLineColor(rt.kBlue)
    TET = f.Get('TET_hist')
    TET.SetLineColor(rt.kBlue)
    jes_TET = f.Get('jes_TET_hist')
    RET_TET = f.Get('RET_TET_hist')
    
    jps = f.Get('jps_hist')
    jps.SetLineColor(rt.kBlue)
    TpT = f.Get('TpT_hist')
    TpT.SetLineColor(rt.kBlue)
    jps_TpT = f.Get('jps_TpT_hist')
    RpT_TpT = f.Get('RpT_TpT_hist')
    # --------------------
    
    # Legend for use by all histograms, without markers
    leg = SetupLegend(0.65,0.775,0.9,0.9)
    leg.SetHeader("anti-k_{T}, R = 0.8")
    leg.AddEntry(pt_j[1],"Hadronic top decays","l")
    leg.AddEntry(pt_j[0],"Light quarks and gluons","l")
    
#    # Legend for use by all histograms, with markers
#    leg2 = SetupLegend(0.5,0.725,0.9,0.9)
#    leg2.SetHeader("anti-k_{T}, R = 0.8")
#    leg2.AddEntry(nx_avg[1],"Hadronic top decays","p")
#    leg2.AddEntry(nx_avg[0],"Light quarks and gluons","p")
    
    # Textbox with information on the dataset
    pave = SetupPave(0.2, 1. - rt.gStyle.GetPadTopMargin(), 1. - rt.gStyle.GetPadRightMargin(),1.)
    pave.SetTextColor(rt.kGray + 2)
    pave.SetTextFont(12) # times-medium-i-normal w/ precision = 2 (scalable & rotatable hardware font)
    pave.SetTextSize(0.04)
    pave.AddText("Top tagging reference dataset (https://zenodo.org/record/2603256)")
    
    # --- Some beautification of plots for display ---
    
    ranges = {}
    ranges['pt'] = ((5.e2,7.e2),(1.e0,1.e7))
    ranges['eta'] = ((-2.5,2.5),(1.e0,1.e6))
    ranges['phi'] = ((-4.,4.),(1.e0,1.e6))
    ranges['et'] = ((5.e2,8.e2),(1.e0,1.e6))
    ranges['m'] = ((0.,1000.),(1.e0,1.e6))
    ranges['n'] = ((0.,202.),(1.,5.e4))
    ranges['nx'] = ((0.,200.),(0.,1.e2))
    ranges['jes'] = ((-1.,1.),(1.,2.e5))
    ranges['TET'] = ((4.e2,1.6e3),(1.e0,1.e6))
    ranges['jes_TET'] = (ranges['TET'][0],(-1.,1.),(0.,1.e5))
    ranges['RET_TET'] = (ranges['TET'][0],ranges['et'][0],(0.,1.e5))
    ranges['jps'] = ranges['jes']
    ranges['TpT'] = ((4.e2,1.6e3),(1.e0,1.e6))
    ranges['jps_TpT'] = (ranges['TpT'][0],(-1.,1.),(0.,1.e5))
    ranges['RpT_TpT'] = (ranges['TpT'][0],ranges['pt'][0],(0.,1.e5))
    ranges['n_m'] = (ranges['m'][0],ranges['n'][0],(0.,1.e5))

    
    [PlotAdjust(pt_j[x], ranges['pt'], ";p_{T} [GeV];Number of jets per 10 GeV") for x in range(2)]
    [PlotAdjust(eta_j[x], ranges['eta'], ";#eta;Number of jets per bin") for x in range(2)]
    [PlotAdjust(phi_j[x], ranges['phi'], ";#phi;Number of jets per bin") for x in range(2)]
    [PlotAdjust(et_j[x], ranges['et'], ";E_{T} [GeV];Number of jets per 10 GeV") for x in range(2)]
    [PlotAdjust(m_j[x], ranges['m'], ";m [GeV];Number of jets per 5 GeV") for x in range(2)]
    [PlotAdjust(nobj[x], ranges['n'], ";N_{particles};Number of jets per bin",True) for x in range(2)]
    [PlotAdjust(nx_e_avg[x], ranges['nx'], ";x (%);#bar{N^{x}_{E}}",True) for x in range(2)]
    [PlotAdjust(nx_et_avg[x], ranges['nx'], ";x (%);#bar{N^{x}_{E_{T}}}",True) for x in range(2)]
    for x in X:
        [PlotAdjust(nx_e[y][x], ranges['n'], ";N^{" + str(x) + "}_{E};Number of jets per bin",True) for y in range(2)]
        [PlotAdjust(nx_et[y][x], ranges['n'], ";N^{" + str(x) + "}_{E_{T}};Number of jets per bin",True) for y in range(2)]

    
    PlotAdjust(n_m, ranges['n_m'], ";m_{j} (truth);N_{particles}")

    PlotAdjust(TET, ranges['TET'], ";E_{T} (truth) [GeV];Number of top quarks per 10 GeV")
    PlotAdjust(jes, ranges['jes'], ";JES - 1;Number of jets per bin")
    PlotAdjust(jes_TET, ranges['jes_TET'], ";E_{T} (truth) [GeV];JES - 1")
    PlotAdjust(RET_TET, ranges['RET_TET'], ";E_{T} (truth) [GeV];E_{T} (reco) [GeV]")
    
    PlotAdjust(TpT, ranges['TpT'], ";p_{T} (truth) [GeV];Number of top quarks per 10 GeV")
    PlotAdjust(jps, ranges['jps'], ";JpS - 1;Number of jets per bin")
    PlotAdjust(jps_TpT, ranges['jps_TpT'], ";p_{T} (truth) [GeV];JpS - 1")
    PlotAdjust(RpT_TpT, ranges['RpT_TpT'], ";p_{T} (truth) [GeV];p_{T} (reco) [GeV]")

    # Draw plots
    canvas_dims = [800,600]
    rt.gStyle.SetOptStat(0)
    c_pt = rt.TCanvas('c_pt','c_pt',canvas_dims[0],canvas_dims[1])
    c_eta = rt.TCanvas('c_eta','c_eta',canvas_dims[0],canvas_dims[1])
    c_phi = rt.TCanvas('c_phi','c_phi',canvas_dims[0],canvas_dims[1])
    c_et = rt.TCanvas('c_et','c_et',canvas_dims[0],canvas_dims[1])
    c_m = rt.TCanvas('c_m','c_m',canvas_dims[0],canvas_dims[1])
    c_n = rt.TCanvas('c_n','c_n',canvas_dims[0],canvas_dims[1])
    c_nx_e_avg = rt.TCanvas('c_nx_e_avg','c_nx_e_avg',canvas_dims[0],canvas_dims[1])
    c_nx_et_avg = rt.TCanvas('c_nx_et_avg','c_nx_et_avg',canvas_dims[0],canvas_dims[1])
    c_nx_e = [rt.TCanvas('c_nx_e_'+str(x),'c_nx_e_'+str(x),canvas_dims[0],canvas_dims[1]) for x in X]
    c_nx_et = [rt.TCanvas('c_nx_et_'+str(x),'c_nx_et_'+str(x),canvas_dims[0],canvas_dims[1]) for x in X]

    c_n_m = rt.TCanvas('c_n_m','c_n_m',canvas_dims[0],canvas_dims[1])
    
    c_TET = rt.TCanvas('c_TET','c_TET',canvas_dims[0],canvas_dims[1])
    c_jes = rt.TCanvas('c_jes','c_jes',canvas_dims[0],canvas_dims[1])
    c_jes_TET = rt.TCanvas('c_jes_TET','c_jes_TET',canvas_dims[0],canvas_dims[1])
    c_RET_TET = rt.TCanvas('c_RET_TET','c_RET_TET',canvas_dims[0],canvas_dims[1])
    
    c_TpT = rt.TCanvas('c_TpT','c_TpT',canvas_dims[0],canvas_dims[1])
    c_jps = rt.TCanvas('c_jps','c_jps',canvas_dims[0],canvas_dims[1])
    c_jps_TpT = rt.TCanvas('c_jps_TpT','c_jps_TpT',canvas_dims[0],canvas_dims[1])
    c_RpT_TpT = rt.TCanvas('c_RpT_TpT','c_RpT_TpT',canvas_dims[0],canvas_dims[1])
    
    PlotDisplay(c_pt, pt_j, leg, pave)
    PlotDisplay(c_eta, eta_j, leg, pave)
    PlotDisplay(c_phi, phi_j, leg, pave)
    PlotDisplay(c_et, et_j, leg, pave)
    PlotDisplay(c_m, m_j, leg, pave)
    PlotDisplay(c_n, nobj, leg, pave, True)
    PlotDisplay(c_nx_e_avg, nx_e_avg, leg, pave, False, False, True)
    PlotDisplay(c_nx_et_avg, nx_et_avg, leg, pave, False, False, True)

    PlotDisplay(c_n_m, [n_m], 0, pave, True, True)

    PlotDisplay(c_TET, [TET], 0, pave)
    PlotDisplay(c_jes, [jes], 0, pave)
    PlotDisplay(c_jes_TET, [jes_TET], 0, pave, True, True)
    PlotDisplay(c_RET_TET, [RET_TET], 0, pave, True, True)
    PlotDisplay(c_TpT, [TpT], 0, pave)
    PlotDisplay(c_jps, [jps], 0, pave)
    PlotDisplay(c_jps_TpT, [jps_TpT], 0, pave, True, True)
    PlotDisplay(c_RpT_TpT, [RpT_TpT], 0, pave, True, True)

    nx_range = range(len(X)-3, len(X))

    [PlotDisplay(c_nx_e[idx], [nx_e[0][X[idx]],nx_e[1][X[idx]]], leg, pave, True) for idx in nx_range]
    [PlotDisplay(c_nx_et[idx], [nx_et[0][X[idx]],nx_et[1][X[idx]]], leg, pave, True) for idx in nx_range]

    g = rt.TFile('plots.root', 'RECREATE')
    CanvasSave(c_pt, 'pt_reco', g)
    CanvasSave(c_eta, 'eta_reco', g)
    CanvasSave(c_phi, 'phi_reco', g)
    CanvasSave(c_et, 'et_reco', g)
    CanvasSave(c_m, 'm_reco', g)
    CanvasSave(c_n, 'n', g)
    CanvasSave(c_nx_e_avg, 'nx_e_avg', g)
    CanvasSave(c_nx_et_avg, 'nx_et_avg', g)
    CanvasSave(c_n_m, 'n_m', g)
    CanvasSave(c_TET, 'et_truth', g)
    CanvasSave(c_jes, 'jes-1', g)
    CanvasSave(c_jes_TET, 'jes-1_et_truth', g)
    CanvasSave(c_RET_TET, 'et_reco_et_truth', g)
    CanvasSave(c_TpT, 'pt_truth', g)
    CanvasSave(c_jps, 'jps-1', g)
    CanvasSave(c_jps_TpT, 'jps-1_pt_truth', g)
    CanvasSave(c_RpT_TpT, 'pt_reco_pt_truth', g)
    
    [CanvasSave(c_nx_e[idx], 'n_e_'+str(int(X[idx])), g) for idx in nx_range]
    [CanvasSave(c_nx_et[idx], 'n_et_'+str(int(X[idx])), g) for idx in nx_range]

    g.Close()
    f.Close()
    return

if __name__ == '__main__':
    main(sys.argv)


