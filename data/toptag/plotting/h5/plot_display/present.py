#  present.py
# Created by Jan Offermann on 01/04/20.
# Goal: Draw the histograms created from the toptag dataset.
#       (Pythonized & updated version of "present.C")

import ROOT as rt, numpy as np, subprocess as sub, sys, os

# Global themes
keynote_blue = rt.TColor.GetColor(52,165,218)
keynote_orange = rt.TColor.GetColor(231,114,61)
keynote_grey = rt.TColor.GetColor(214,213,213)
keynote_black = rt.TColor.GetColor(34,34,34)

themes = {
'light':{'main':rt.kBlack,'canvas':rt.kWhite,'text':rt.kBlack, 'pave':rt.kGray + 2,'signal':rt.kBlue,'background':rt.kRed},
'dark':{'main':keynote_blue,'canvas':keynote_black, 'text':rt.kWhite, 'pave':rt.kWhite, 'signal':keynote_orange,'background':keynote_grey}
}

# Helper function for making TPaveText
def SetupPave(x1, y1, x2, y2, NDC = True, mode = 'light'):
    option = 'NDC'
    if(not NDC): option = ''
    pave = rt.TPaveText(x1,y1,x2,y2, option)
    pave.SetTextFont(42)
    pave.SetTextAngle(0)
    pave.SetTextColor(themes[mode]['text'])
    pave.SetTextSize(0.04)
    pave.SetTextAlign(12)
    pave.SetFillStyle(0)
    pave.SetBorderSize(0)
    return pave

def SetupLegend(x1, y1, x2, y2, mode = 'light'):
    leg = rt.TLegend(x1,y1,x2,y2)
    leg.SetFillColor(0)
    leg.SetFillStyle(0) # make it transparent!
    leg.SetBorderSize(0)
    leg.SetTextFont(42)
    leg.SetTextAngle(0)
    leg.SetTextColor(themes[mode]['text'])
    leg.SetTextSize(0.03)
    leg.SetTextAlign(12)
    return leg

def PlotAdjust(histogram, ranges, title = '', scientific = False, mode = 'light'):
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
            
    # set colors of axes
    for i in range(naxes):
        axes[i].SetTitleColor(themes[mode]['text'])
        axes[i].SetLabelColor(themes[mode]['text'])
    return

def PlotDisplay(c, plots, leg, paves, logy = False, logz = False, d2 = False, d3 = False, errors = False, option = '',mode='light'):
    nplots = len(plots)
    c.cd()
    rt.gPad.SetFillColorAlpha(rt.kWhite,0.)
    if(option == ''):
        if(d2): option = 'COLZ'
        if(d3): option = 'SURF7'
        if(errors): option = 'HIST E1'
        
    if(option == 'COLZ'): c.SetRightMargin(0.125) # leave space for colorbar labels (sufficient for log scale)
    
    if('CANDLE' in option and nplots == 2): # TODO: Quick hack for box and whisker plot. Maybe make this nicer?
        plots[0].SetBarWidth(0.3)
        plots[0].SetBarOffset(-0.20)
        plots[1].SetBarWidth(0.3)
        plots[1].SetBarOffset(0.20)

    # Draw the last plot in the list. (by convention, this is the background plot)
    if('SURF' in option and nplots == 2): rt.gStyle.SetPalette(rt.kCherry)
    plots[-1].Draw(option)
    if('SURF' in option and nplots == 2): rt.gStyle.SetPalette(rt.kDeepSea)

    for i in range(nplots-2,-1,-1):
        if(option == ''): option = 'SAME'
        else: option += ' SAME'
        plots[i].Draw(option)
    
    if(leg != 0): leg.Draw()
    for pave in paves: pave.Draw()
    if(logy): c.SetLogy()
    if(logz): c.SetLogz()
        
    # if using linear scale with 2D/3D plot, adjust maximum cleverly TODO: Do this for all plots?
    if(not (logy or logz) and (d2 or d3)):
        max = plots[-1].GetBinContent(plots[-1].GetMaximumBin())
        max_order = np.floor(np.log10(max))
        max_order = np.power(10, max_order)
        max = np.ceil(max/max_order) * max_order
        plots[-1].GetZaxis().SetRangeUser(0.,max)
    
    # reduce font sizes for 3D plots (those using SURF option)
    if(d3):
        label_scale = 0.5
        plots[-1].GetXaxis().SetLabelSize(label_scale * plots[-1].GetXaxis().GetLabelSize())
        plots[-1].GetYaxis().SetLabelSize(label_scale * plots[-1].GetYaxis().GetLabelSize())
        plots[-1].GetZaxis().SetLabelSize(label_scale * plots[-1].GetZaxis().GetLabelSize())
    
    # again force the axis colors TODO: Why is this necessary again? Was set in gStyle but only colors half the axes...
    for plot in plots:
        plot.GetXaxis().SetAxisColor(themes[mode]['main'])
        plot.GetYaxis().SetAxisColor(themes[mode]['main'])
        plot.GetZaxis().SetAxisColor(themes[mode]['main'])

    c.Draw()
    rt.gPad.Update()
    rt.gPad.RedrawAxis()
#    c.RedrawAxis() # redraw axis, in case there is any overlap
#    c.Update()
    return

def CanvasSave(c, prefix, f, suffixes = ['png','eps','pdf'],mode='light'):
    # adjust the background color of the canvas
    c.SetFillColor(themes[mode]['canvas'])
    for suffix in suffixes:
        c.SaveAs(prefix + '.' + suffix)
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

def GetXYLine(plot_range):
    x1 = plot_range[0][0]
    y1 = plot_range[1][0]
    x2 = plot_range[0][1]
    y2 = plot_range[1][1]
    if(y1 > x1): x1 = y1
    else: y1 = x1
    if(y2 > x2): y2 = x2
    else: x2 = y2
    line = rt.TLine(x1,y1,x2,y2)
    line.SetLineColor(rt.kRed)
    line.SetLineWidth(2)
    line.SetLineStyle(9)
    return line

def main(args):
    if(len(sys.argv) > 1): filename = str(sys.argv[1])
    else: filename = 'toptag.root'
    
    if(len(sys.argv) > 2): mode = str(sys.argv[2])
    else: mode = 'light'
    if(mode not in list(themes.keys())):
        print('Error: Mode (arg2) not understood. Current options are ',list(themes.keys()))
        return
    
    if(len(sys.argv) > 3): cut = str(sys.argv[3])
    else: cut = ''
    
    # Some shorthand options for cuts
#    if(cut == 'mj'): cut = 'm_{j}#in(m_{top}- 50[GeV], m_{top}+ 50[GeV])'
    if(cut == 'mj'): cut = 'm_{j}#in(m_{top}#pm 50[GeV])'
    
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
    
    # set the style TODO: Setting this doesn't always seem to work, so we explicitly set some colors throughout
    rt.gStyle.SetTitleTextColor(themes[mode]['text'])
    rt.gStyle.SetLabelColor(themes[mode]['text'])
    rt.gStyle.SetGridColor(themes[mode]['main'])
    rt.gStyle.SetAxisColor(themes[mode]['main'])
    
    f = rt.TFile(filename,'READ')
    suffixes = ['sig','bck']
    suffixes2 = ['sig_bck','all'] # for some vars, we have sig & background series, and combined series
    colors = {suffixes[0] : themes[mode]['signal'], suffixes[1] : themes[mode]['background']}

    # Jet histograms
    pt_j = [f.Get('pt_j_hist_'+x) for x in suffixes]
    [pt_j[x].SetLineColor(colors[suffixes[x]]) for x in range(2)]
    
    eta_j = [f.Get('eta_j_hist_'+x) for x in suffixes]
    [eta_j[x].SetLineColor(colors[suffixes[x]]) for x in range(2)]

    phi_j = [f.Get('phi_j_hist_'+x) for x in suffixes]
    [phi_j[x].SetLineColor(colors[suffixes[x]]) for x in range(2)]

    et_j = [f.Get('et_j_hist_'+x) for x in suffixes]
    [et_j[x].SetLineColor(colors[suffixes[x]]) for x in range(2)]

    m_j = [f.Get('m_j_hist_'+x) for x in suffixes]
    [m_j[x].SetLineColor(colors[suffixes[x]]) for x in range(2)]

    nobj = [f.Get('n_hist_'+x) for x in suffixes]
    [nobj[x].SetLineColor(colors[suffixes[x]]) for x in range(2)]

    X = np.linspace(0,100,21,dtype=np.dtype('i8')) #TODO: This must match the array in plotting_h5.py
    nx_e = [{x: f.Get('n'+str(x)+'_e_hist_'+y) for x in X} for y in suffixes]
    [[nx_e[y][x].SetLineColor(colors[suffixes[y]]) for y in range(2)] for x in X]

    nx_et = [{x: f.Get('n'+str(x)+'_et_hist_'+y) for x in X} for y in suffixes]
    [[nx_et[y][x].SetLineColor(colors[suffixes[y]]) for y in range(2)] for x in X]

    # Make the average Nx distributions, using the Nx distributions above
    nx_e_avg = AvgNx(f,X, suffixes, False)
    [nx_e_avg[x].SetLineColor(colors[suffixes[x]]) for x in range(2)]
    [nx_e_avg[x].SetMarkerColor(colors[suffixes[x]]) for x in range(2)]
    
    nx_et_avg = AvgNx(f,X, suffixes, True)
    [nx_et_avg[x].SetLineColor(colors[suffixes[x]]) for x in range(2)]
    [nx_et_avg[x].SetMarkerColor(colors[suffixes[x]]) for x in range(2)]
    # --------------------
    
    e_frac = [f.Get('frac_E_hist_'+x) for x in suffixes]
    [e_frac[x].SetLineColor(colors[suffixes[x]]) for x in range(2)]

    # For 2D histograms with separate signal/background, also make a combo
    n_m  = [f.Get('n_m_hist_'+x)  for x in suffixes] # N vs m_j histograms
    n_m.append(n_m[1].Clone('n_m_hist_all'))
    n_m[-1].Add(n_m[0])
    [n_m[x].SetLineColor(colors[suffixes[x]]) for x in range(2)]

    n_pt = [f.Get('n_pt_hist_'+x) for x in suffixes] # N vs pt_reco histograms
    n_pt.append(n_pt[1].Clone('n_pt_hist_all'))
    n_pt[-1].Add(n_pt[0])
    [n_pt[x].SetLineColor(colors[suffixes[x]]) for x in range(2)]
    
    e_frac_pt = [f.Get('frac_E_pt_hist_'+x) for x in suffixes] # N vs pt_reco histograms
    e_frac_pt.append(e_frac_pt[1].Clone('frac_E_pt_hist_all'))
    e_frac_pt[-1].Add(e_frac_pt[0])
    [e_frac_pt[x].SetLineColor(colors[suffixes[x]]) for x in range(2)]

    n_TpT = f.Get('n_TpT_hist') # N vs pt_true histogram
    
    # Jet energy scale & other signal-only histograms
    jes = f.Get('jes_hist')
    jes.SetLineColor(themes[mode]['signal'])
    TET = f.Get('TET_hist')
    TET.SetLineColor(themes[mode]['signal'])
    jes_TET = f.Get('jes_TET_hist')
    RET_TET = f.Get('RET_TET_hist')
    
    jps = f.Get('jps_hist')
    jps.SetLineColor(themes[mode]['signal'])
    TpT = f.Get('TpT_hist')
    TpT.SetLineColor(themes[mode]['signal'])
    jps_TpT = f.Get('jps_TpT_hist')
    RpT_TpT = f.Get('RpT_TpT_hist')
    # --------------------
    
    # Legend for use by all histograms, without markers
    leg_x1 = 0.65
    leg_x2 = 0.9
    leg_y1 = 0.775
    leg_y2 = 0.9
    leg = SetupLegend(leg_x1, leg_y1, leg_x2, leg_y2,mode=mode)
    leg.SetHeader("anti-k_{T}, R = 0.8")
    leg.AddEntry(pt_j[0],"Hadronic top decays","l")
    leg.AddEntry(pt_j[1],"Light quarks and gluons","l")
    
    paves = []
    # Textbox with information on the dataset
    pave = SetupPave(0.2, 1. - rt.gStyle.GetPadTopMargin(), 1. - rt.gStyle.GetPadRightMargin(),1.,mode=mode)
    pave.SetTextColor(themes[mode]['pave'])
    pave.SetTextFont(12) # times-medium-i-normal w/ precision = 2 (scalable & rotatable hardware font)
    pave.SetTextSize(0.04)
    pave.AddText('Top tagging reference dataset (https://zenodo.org/record/2603256)')
    paves.append(pave)
    
    # Optional textbox with cut information (none if there's not cut
    pave2 = 0
    if(cut != ''):
        pave2 = SetupPave(0.1, 0., 1. - rt.gStyle.GetPadRightMargin(),rt.gStyle.GetPadBottomMargin(),mode=mode)
        pave2.SetTextColor(rt.kViolet + 3)
        pave2.SetTextFont(12) # times-medium-i-normal w/ precision = 2 (scalable & rotatable hardware font)
        pave2.SetTextSize(0.03)
        pave2.AddText(cut)
        paves.append(pave2)
    
    # --- Some beautification of plots for display ---
    ranges = {}
    ranges['pt'] = ((5.5e2,6.5e2),(1.e0,1.e7))
    ranges['eta'] = ((-2.5,2.5),(1.e0,1.e6))
    ranges['phi'] = ((-4.,4.),(1.e0,1.e6))
    ranges['et'] = ((5.e2,8.e2),(1.e0,1.e6))
    ranges['m'] = ((0.,400.),(1.e0,1.e6))
    ranges['n'] = ((0.,202.),(1.,5.e4))
    ranges['nx'] = ((0.,200.),(0.,5.e1))
    ranges['e_frac'] = ((1.e-6,100.),(1.e-1,1.e7))
    ranges['jes'] = ((-1.,1.),(1.,2.e5))
    ranges['TET'] = ((4.e2,1.6e3),(1.e0,1.e6))
    ranges['jes_TET'] = (ranges['TET'][0],(-1.,1.),(0.,1.e5))
    ranges['RET_TET'] = (ranges['TET'][0],ranges['et'][0],(0.,1.e5))
    ranges['jps'] = ranges['jes']
    ranges['TpT'] = ((4.e2,1.6e3),(1.e0,1.e6))
    ranges['jps_TpT'] = (ranges['TpT'][0],(-1.,1.),(0.,1.e5))
    ranges['RpT_TpT'] = (ranges['TpT'][0],ranges['pt'][0],(0.,1.e5))
    ranges['n_m'] = (ranges['m'][0],ranges['n'][0],(0.,1.e5))
    ranges['n_pt'] = (ranges['pt'][0],ranges['n'][0], (0.,1.e5))
    ranges['e_frac_pt'] = (ranges['pt'][0],(1.e-3,5.e0), (1.e4,1.e7))
    ranges['n_TpT'] = (ranges['TpT'][0],ranges['n'][0], (0.,1.e5))

    [PlotAdjust(pt_j[x], ranges['pt'], ";p_{T} [GeV];Number of jets per 10 GeV",mode=mode) for x in range(2)]
    [PlotAdjust(eta_j[x], ranges['eta'], ";#eta;Number of jets per bin",mode=mode) for x in range(2)]
    [PlotAdjust(phi_j[x], ranges['phi'], ";#phi;Number of jets per bin",mode=mode) for x in range(2)]
    [PlotAdjust(et_j[x], ranges['et'], ";E_{T} [GeV];Number of jets per 10 GeV",mode=mode) for x in range(2)]
    [PlotAdjust(m_j[x], ranges['m'], ";m_{j} [GeV];Number of jets per 5 GeV",mode=mode) for x in range(2)]
    [PlotAdjust(nobj[x], ranges['n'], ";N_{constituents};Number of jets per bin",True,mode=mode) for x in range(2)]
    [PlotAdjust(nx_e_avg[x], ranges['nx'], ";x (%);#bar{N^{x}_{E}}",True,mode=mode) for x in range(2)]
    [PlotAdjust(nx_et_avg[x], ranges['nx'], ";x (%);#bar{N^{x}_{E_{T}}}",True,mode=mode) for x in range(2)]
    [PlotAdjust(e_frac[x], ranges['e_frac'], ";% Jet Energy;Number of jet constituents per bin",True,mode=mode) for x in range(2)]
    [PlotAdjust(n_m[x], ranges['n_m'], ";m_{j} [GeV];N_{particles}",mode=mode) for x in range(3)] # Note loop over 3 entries
    [PlotAdjust(n_pt[x], ranges['n_pt'], ";p_{T}^{reco} [GeV];N_{particles}",mode=mode) for x in range(3)] # Note loop over 3 entries
    [PlotAdjust(e_frac_pt[x], ranges['e_frac_pt'], ";p_{T}^{reco} [GeV];% Jet Energy",mode=mode) for x in range(3)] # Note loop over 3 entries
    
    for x in X:
        [PlotAdjust(nx_e[y][x], ranges['n'], ";N^{" + str(x) + "}_{E};Number of jets per bin",True,mode=mode) for y in range(2)]
        [PlotAdjust(nx_et[y][x], ranges['n'], ";N^{" + str(x) + "}_{E_{T}};Number of jets per bin",True,mode=mode) for y in range(2)]
    
    PlotAdjust(n_TpT, ranges['n_TpT'], ";p_{T}^{truth} [GeV];N_{particles}",mode=mode)
    
    PlotAdjust(TET, ranges['TET'], ";E_{T}^{truth} [GeV];Number of top quarks per 10 GeV",mode=mode)
    PlotAdjust(jes, ranges['jes'], ";JES - 1;Number of jets per bin",mode=mode)
    PlotAdjust(jes_TET, ranges['jes_TET'], ";E_{T}^{truth} [GeV];JES - 1",mode=mode)
    PlotAdjust(RET_TET, ranges['RET_TET'], ";E_{T}^{truth} [GeV];E_{T}^{reco} [GeV]",mode=mode)
    
    PlotAdjust(TpT, ranges['TpT'], ";p_{T} (truth) [GeV];Number of top quarks per 10 GeV",mode=mode)
    PlotAdjust(jps, ranges['jps'], ";JpS - 1;Number of jets per bin",mode=mode)
    PlotAdjust(jps_TpT, ranges['jps_TpT'], ";p_{T}^{truth} [GeV];JpS - 1",mode=mode)
    PlotAdjust(RpT_TpT, ranges['RpT_TpT'], ";p_{T}^{truth} [GeV];p_{T}^{reco} [GeV]",mode=mode)

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
    c_e_frac = rt.TCanvas('c_e_frac','c_e_frac',canvas_dims[0],canvas_dims[1])
    
    # 2D histograms
    c_n_m = [rt.TCanvas('c_n_m_'+x,'c_n_m_'+x,canvas_dims[0],canvas_dims[1]) for x in suffixes2]
    c_n_pt = [rt.TCanvas('c_n_pt_'+x,'c_n_pt_'+x,canvas_dims[0],canvas_dims[1]) for x in suffixes2]
    c_e_frac_pt = [rt.TCanvas('c_e_frac_pt_'+x,'c_e_frac_pt_'+x,canvas_dims[0],canvas_dims[1]) for x in suffixes2]
    c_n_TpT = rt.TCanvas('c_n_TpT','c_n_TpT',canvas_dims[0],canvas_dims[1])
    
    c_n_m_3d = rt.TCanvas('c_n_m_3d','c_n_m_3d',canvas_dims[0],canvas_dims[1])
    c_n_pt_3d = rt.TCanvas('c_n_pt_3d','c_n_pt_3d',canvas_dims[0],canvas_dims[1])
    c_n_TpT_3d = rt.TCanvas('c_n_TpT_3d','c_n_TpT_3d',canvas_dims[0],canvas_dims[1])

    c_TET = rt.TCanvas('c_TET','c_TET',canvas_dims[0],canvas_dims[1])
    c_jes = rt.TCanvas('c_jes','c_jes',canvas_dims[0],canvas_dims[1])
    c_jes_TET = rt.TCanvas('c_jes_TET','c_jes_TET',canvas_dims[0],canvas_dims[1])
    c_RET_TET = rt.TCanvas('c_RET_TET','c_RET_TET',canvas_dims[0],canvas_dims[1])
    
    c_TpT = rt.TCanvas('c_TpT','c_TpT',canvas_dims[0],canvas_dims[1])
    c_jps = rt.TCanvas('c_jps','c_jps',canvas_dims[0],canvas_dims[1])
    c_jps_TpT = rt.TCanvas('c_jps_TpT','c_jps_TpT',canvas_dims[0],canvas_dims[1])
    c_RpT_TpT = rt.TCanvas('c_RpT_TpT','c_RpT_TpT',canvas_dims[0],canvas_dims[1])
    
    PlotDisplay(c_pt, pt_j, leg, paves,logy=True,mode=mode)
    PlotDisplay(c_eta, eta_j, leg, paves,logy=True,mode=mode)
    PlotDisplay(c_phi, phi_j, leg, paves,logy=True,mode=mode)
    PlotDisplay(c_et, et_j, leg, paves,logy=True,mode=mode)
    PlotDisplay(c_m, m_j, leg, paves,logy=True,mode=mode)
    PlotDisplay(c_n, nobj, leg, paves,logy=True,mode=mode)
    PlotDisplay(c_nx_e_avg, nx_e_avg, leg, paves, errors=True,mode=mode)
    PlotDisplay(c_nx_et_avg, nx_et_avg, leg, paves, errors=True,mode=mode)
    PlotDisplay(c_e_frac, e_frac, leg, paves,logy=True,mode=mode)

    # 2D histograms -- for 2D representations of n vs. m and n vs. pt, we use candle plots for separating signal and background
    draw_options = ['CANDLEX3','COLZ']
    PlotDisplay(c_n_m[0], n_m[:2], leg, paves, logz=True, d2=True, option=draw_options[0],mode=mode)
    PlotDisplay(c_n_m[1], [n_m[2]], 0, paves, logz=True, d2=True, option=draw_options[1],mode=mode)
    PlotDisplay(c_n_pt[0], n_pt[:2], leg, paves, logz=True, d2=True, option=draw_options[0],mode=mode)
    PlotDisplay(c_n_pt[1], [n_pt[2]], 0, paves, logz=True, d2=True, option=draw_options[1],mode=mode)
    PlotDisplay(c_e_frac_pt[0], e_frac_pt[:2], leg, paves, logy=False, logz=True, d2=True, option=draw_options[0],mode=mode)
    PlotDisplay(c_e_frac_pt[1], [e_frac_pt[2]], 0, paves, logy=False, logz=True, d2=True, option=draw_options[1],mode=mode)
    PlotDisplay(c_n_TpT, [n_TpT], 0, paves, logz=True, d2=True,mode=mode)
    
    PlotDisplay(c_n_m_3d, [n_m[2]], 0, paves, d3=True,mode=mode)
    PlotDisplay(c_n_pt_3d, [n_pt[2]], 0, paves, d3=True,mode=mode)
    PlotDisplay(c_n_TpT_3d, [n_TpT], 0, paves, d3=True,mode=mode)

    PlotDisplay(c_TET, [TET], 0, paves, logy=True,mode=mode)
    PlotDisplay(c_jes, [jes], 0, paves, logy=True,mode=mode)
    PlotDisplay(c_jes_TET, [jes_TET], 0, paves, logz=True, d2=True,mode=mode)
    PlotDisplay(c_RET_TET, [RET_TET], 0, paves, logz=True, d2=True,mode=mode)
    PlotDisplay(c_TpT, [TpT], 0, paves, logy=True,mode=mode)
    PlotDisplay(c_jps, [jps], 0, paves, logy=True,mode=mode)
    PlotDisplay(c_jps_TpT, [jps_TpT], 0, paves, logz=True, d2=True,mode=mode)
    PlotDisplay(c_RpT_TpT, [RpT_TpT], 0, paves, logz=True, d2=True,mode=mode)
    
    # get y=x line on the RpT_TpT plot
    line = GetXYLine(ranges['RpT_TpT'])
    c_RpT_TpT.cd()
    line.Draw()
    # label the line
    offset = 0.95
    label_x = offset * line.GetX2()
    label_y = offset * line.GetY2()
    label_scale_x = 200.
    label_scale_y = 20.
    line_label = SetupPave(1.05 * label_x,label_y, label_x + label_scale_x, label_y + label_scale_y, False,mode=mode)
    line_label.SetTextColor(rt.kRed)
    line_label.AddText('p_{T, reco} = p_{T, truth}')
    line_label.Draw()

    nx_range = range(len(X)-3, len(X))
    [PlotDisplay(c_nx_e[idx], [nx_e[0][X[idx]],nx_e[1][X[idx]]], leg, paves, logy=True,mode=mode) for idx in nx_range]
    [PlotDisplay(c_nx_et[idx], [nx_et[0][X[idx]],nx_et[1][X[idx]]], leg, paves, logy=True,mode=mode) for idx in nx_range]

    g = rt.TFile('plots.root', 'RECREATE')
    file_suffixes = ['png','eps','pdf']
    
    CanvasSave(c_pt, 'pt_reco', g, file_suffixes,mode=mode)
    CanvasSave(c_eta, 'eta_reco', g, file_suffixes,mode=mode)
    CanvasSave(c_phi, 'phi_reco', g, file_suffixes,mode=mode)
    CanvasSave(c_et, 'et_reco', g, file_suffixes,mode=mode)
    CanvasSave(c_m, 'm_reco', g, file_suffixes,mode=mode)
    
    # Save log & linear version of the n plot
    CanvasSave(c_n, 'n_log', g, file_suffixes,mode=mode)
    c_n.SetLogy(0)
    CanvasSave(c_n, 'n', g, file_suffixes,mode=mode)
    
    CanvasSave(c_nx_e_avg, 'nx_e_avg', g, file_suffixes,mode=mode)
    CanvasSave(c_nx_et_avg, 'nx_et_avg', g, file_suffixes,mode=mode)
    CanvasSave(c_e_frac, 'e_frac', g, file_suffixes,mode=mode)

    [CanvasSave(c_n_m[x], 'n_m_'+suffixes2[x], g, file_suffixes,mode=mode) for x in range(2)]
    [CanvasSave(c_n_pt[x], 'n_pt_'+suffixes2[x], g, file_suffixes,mode=mode) for x in range(2)]
    [CanvasSave(c_e_frac_pt[x], 'e_frac_pt_'+suffixes2[x], g, file_suffixes,mode=mode) for x in range(2)]
    CanvasSave(c_n_TpT, 'n_TpT', g, file_suffixes,mode=mode)
    
    CanvasSave(c_n_m_3d, 'n_m_3d', g, file_suffixes,mode=mode)
    CanvasSave(c_n_pt_3d, 'n_pt_3d', g, file_suffixes,mode=mode)
    CanvasSave(c_n_TpT_3d, 'n_TpT_3d', g, file_suffixes,mode=mode)
    
    CanvasSave(c_TET, 'et_truth', g, file_suffixes,mode=mode)
    CanvasSave(c_jes, 'jes-1', g, file_suffixes,mode=mode)
    CanvasSave(c_jes_TET, 'jes-1_et_truth', g, file_suffixes,mode=mode)
    CanvasSave(c_RET_TET, 'et_reco_et_truth', g, file_suffixes,mode=mode)
    CanvasSave(c_TpT, 'pt_truth', g, file_suffixes,mode=mode)
    CanvasSave(c_jps, 'jps-1', g, file_suffixes,mode=mode)
    CanvasSave(c_jps_TpT, 'jps-1_pt_truth', g, file_suffixes,mode=mode)
    CanvasSave(c_RpT_TpT, 'pt_reco_pt_truth', g, file_suffixes,mode=mode)
    
    [CanvasSave(c_nx_e[idx], 'n_e_'+str(int(X[idx])), g, file_suffixes,mode=mode) for idx in nx_range]
    [CanvasSave(c_nx_et[idx], 'n_et_'+str(int(X[idx])), g, file_suffixes,mode=mode) for idx in nx_range]

    g.Close()
    f.Close()
    
    # place the plots in subdirectories based on filetype
    for suffix in file_suffixes:
        try: os.makedirs(suffix)
        except: pass
        sub.check_call('mv *.' + suffix + ' ' + suffix + '/', shell=True)
    return

if __name__ == '__main__':
    main(sys.argv)
