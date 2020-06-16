# Plotting from raw files

import numpy as np, h5py as h5, ROOT as rt
import sys, glob

def SetupPave(x1,y1,x2,y2,use_NDC = False):
    option = ""
    if(use_NDC): option = "NDC"
    pave = rt.TPaveText(x1,y1,x2,y2,option)
    pave.SetTextFont(42)
    pave.SetTextAngle(0)
    pave.SetTextColor(rt.kBlack)
    pave.SetTextSize(0.04)
    pave.SetTextAlign(12)
    pave.SetFillStyle(0)
    pave.SetBorderSize(0)
    return pave

def main(args):
    file_dir = str(sys.argv[1])
    event_index = int(sys.argv[2]) # index of event to plot
    files = glob.glib(file_dir + '/*.h5')
    files = [h5.File(x,'r') for x in files]
    keys = ['Nobj','Pmu','is_signal']
    npar = 200
    nevents = np.array([x['Nobj'].shape[0] for x in files],np.dtype('i8'))
    nevents_total = int(np.sum(nevents))
    
    # Event display histogram
    display = rt.TH2F('event_disp',';#eta;#phi;p_{T} [GeV]',100,-2.,2.,100,-1. * np.pi, np.pi)
    
    # Determine which event to select, from which file
    if event_index >= nevents_total: event_index = nevents_total - 1
    file_index = 0
    for idx in range(nevents.shape[0]):
        if event_index < nevents[idx]:
            file_index = idx
            break
    event_index = event_index - np.sum(nevents[0:file_index])

    file = files[file_index]
    Nobj = file['Nobj'][event_index]
    sig = files[file_index]['is_signal'][event_index]
    pmu = file['Pmu'][event_index,:Nobj,:]
    
    vec = rt.TLorentzVector()
    for i in range(Nobj):
        vec.SetPxPyPzE(pmu[i,1],pmu[i,2],pmu[i,3],pmu[i,0])
        display.Fill(vec.Eta(),vec.Phi(),vec.Pt())
       
       
    # Beautification of histogram
    label_size = 0.05
    title_size = 0.05
    title_offset_x = 0.875
    title_offset_y = 0.7
    title_offset_z = 1.
    
    display.GetXaxis().SetLabelSize(label_size)
    display.GetYaxis().SetLabelSize(label_size)
    display.GetZaxis().SetLabelSize(0.04)
    display.GetXaxis().SetTitleSize(title_size)
    display.GetYaxis().SetTitleSize(title_size)
    display.GetZaxis().SetTitleSize(title_size)
    display.GetXaxis().SetTitleOffset(title_offset_x)
    display.GetYaxis().SetTitleOffset(title_offset_y)
    display.GetZaxis().SetTitleOffset(title_offset_z)
    
    

    # Make a TPaveText with dataset info
    text = 'Top tagging reference dataset (arXiv:1902.09914)'
    pave = SetupPave(-0.7,np.pi,2.,np.pi + 0.4)
    pave.AddText(text)
    pave.SetTextColor(rt.kGray + 2)
    pave.SetTextFont(12)

    # Make a TPaveText showing whether this is signal or background
    text = ['Light quark / gluon','Hadronic top decay'] # [background,signal]
    text = text[sig]
    legend = SetupPave(0.5,2.,2.,np.pi) # coordinates are not in NDC mode -- they are relative to axes' values
    legend.AddText(text)
    
    # Make a TCanvas to draw everything on
    canvas = rt.TCanvas('c1','c1',800,600)
    canvas.cd()
    canvas.SetLeftMargin(0.075)
    canvas.SetRightMargin(0.15)
    display.Draw('COLZ')
    legend.Draw()
    pave.Draw()
    canvas.Draw()
    rt.gPad.Update()
    
    # Adjust the TPaletteAxis object made with the `COLZ` option for TH2F
    palette = display.GetListOfFunctions().FindObject('palette')
    palette.SetX2(2.15)
    
    # Adjustments to the z-axis
    for i in range(7): # hard-coding label numbers (appears to use 1-indexing, odd choice for ROOT)
        display.GetZaxis().ChangeLabel(i+1,30.)
    
    canvas.Update()
    
    # Make TFile, save everything in it
    tfile = rt.TFile('event_disp.root','RECREATE')
    display.Write()
    canvas.Write()
    tfile.Close()
    
    for file in files:
        file.close()

if __name__ == '__main__':
    main(sys.argv)

