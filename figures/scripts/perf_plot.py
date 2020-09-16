import warnings
warnings.filterwarnings("ignore")
import numpy as np, subprocess as sub, matplotlib as mpl
mpl.use('Agg') # batch mode
import matplotlib.pyplot as plt
import os, sys, glob, re
from compress import MakeMiniLog, CheckTimestamp

# Check for ROOT: We'll make ROOT plots
# (together with matplotlib) if possible.
use_ROOT = True
try: import ROOT as rt
except:
    print('Warning: PyROOT is not set up. Disabling ROOT plotting, will only use matplotlib.')
    use_ROOT = False
    
# Global variables
type_conversion = {
'train': 'Training',
'valid': 'Validation'
}
key_conversion = {
'loss': ('Loss','Loss (%)',-4),
'acc': ('Accuracy','Accuracy (%)',-3),
'auc': ('AUC','AUC (%)',-2),
'rej': ('Background rejection @ 0.3','Background Rejection @ 30% Signal Efficiency',-1)
}
axis_range = { #y_min, y_max, ndivisions
'loss': (15,25,10),
'acc': (90,95,10), #(90,95,5)
'auc': (76,100,12),
'rej': (0,800,16)
}

# Sort lists alphanumerically
def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

# Pythonic grep (a.k.a poor man's grep -- might be slower than *nix grep)
def grep(filename, exp):
    lines = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            if exp in line:
                lines.append(line)
    return lines

def hex_to_rgb(value):
    value = value.lstrip('#')
    return list(int(value[i:i+2], 16) for i in (0, 2, 4))
  
def GetData(run_folder, nepochs = 10, type_key = 'train'):

    # Get a name for this run -- will be named after the run_folder.
    name = run_folder.split('/')[-1]

    # Get the log file for the given run.
    logs = glob.glob(run_folder + '/*.log')
    if(len(logs) > 1): # too many logs
        print('Error: Multiple log files found in ' + run_folder + '.')
        return ['err',{0:0}]
    elif(len(logs) > 0): # one log found
        filename = logs[0]
        # Check if minilog is made
        minilog = filename.replace('.log','.minilog')
        if(not os.path.isfile(minilog)):
            print('No minilog found for ', filename,', creating one... (This will speed up plot regeneration.)')
            MakeMiniLog(filename)
        # Check if minilog is up-to-date.
        if (not CheckTimestamp(filename,minilog)):
            print('Minilog for ', filename,' is out of date, creating new one...')
            MakeMiniLog(filename)
        filename = minilog
    else: # no log found -- OK *if* minilog already exists
        print('Warning: No log found in ' + run_folder + '. Looking for existing minilog...')
        logs = glob.glob(run_folder + '/*.minilog')
        if(len(logs) > 1): # too many minilogs
            print('Error: Multiple minilog files found in ' + run_folder + '.')
            return ['err',{0:0}]
        elif(len(logs) == 0): # no minilogs -- this is a problem
            print('Error: No log or minilog file found in ' + run_folder + '.')
            return ['err',{0:0}]
        else: filename = logs[0]
 
    train_string = 'Epoch XXX Complete! Current Training Loss:'
    valid_string = 'Epoch XXX Complete! Current Validation Loss:'
    match_string = {
    'train': train_string,
    'valid': valid_string
    }
    data = {}
    for i in range(nepochs):
        epoch = i + 1
        lines = grep(filename,match_string[type_key].replace('XXX',str(epoch)))
        if not lines:
            data[epoch] = 'MISSING'
            print('Warning: No data found for type ', type_key, ', epoch ', str(epoch), ' in file ', filename, ' .')
            continue
        data[epoch] = lines[-1].split('@')[0].split()[-4:]
    return [type_key, data, name]

def GetAvgData(run_folders, nepochs = 10, type_key='train'):
    nruns = len(run_folders)
    data = [GetData(run_folder,nepochs,type_key)[1] for run_folder in run_folders]
    # Need the minimum # of epochs - the different runs may have made different amounts of progress
    nepochs = np.min(np.array([len(x) for x in data]))
    # Now get avg. metric per epoch & RMS
    avg_data = {}
    for i in range(nepochs):
        # Note: Some data may be missing from log files.
        #       We need to account for this possibility
        #       -> we'll avoid one-liners in favor of some
        #       more careful code.
        epoch_data = []
        for j in range(nruns):
            if (data[j][i+1] != 'MISSING'): epoch_data.append(np.array(data[j][i+1],dtype=np.dtype('f8')))
        nruns_epoch = float(len(epoch_data)) # may be shorter than nruns if data is missing
        avg = np.sum(np.array(epoch_data),axis=0) / nruns_epoch
        stdev = np.std(np.array(epoch_data),axis=0,ddof=1) # ddof=1 since this is the corrected *sample* standard deviation (Bessel's correction)
        stderr = stdev / np.sqrt(nruns_epoch)
        avg_data[i+1] = (avg,stderr)
    return [type_key, avg_data, nepochs]

  # Make plots for of {loss, ACC, AUC, rejection} for each run listed in data_list, with all listed runs on the same axis.
  
def Plot(data_list,key, savedir = '', title = '', overwrite = False):
    # make sure input format is a list, even if single entry
    if(type(data_list) != list): data_list = [data_list]
    
    # mpl setup
    fig = plt.figure()
    ax = fig.add_subplot(111) # why do we need to do these things in matplotlib?

    # ROOT setup
    if(use_ROOT): canv = rt.TCanvas('c1','c1',800,600)
    
    # general setup: colors
    colors = ['xkcd:kelly green','xkcd:aqua blue', 'xkcd:true blue','xkcd:medium purple','xkcd:macaroni and cheese', 'xkcd:burnt orange','xkcd:brick red','xkcd:dark teal','xkcd:bright red']
    colors_hex = ['#02ab2e','#02d8e9','#010fcc','#9e43a2', '#efb435', '#c04e01', '#8f1402', '#014d4e', '#ff000d'] # TODO: Use a single list (colors_hex is more practical, but xkcd names are more human-readable)
    
    # general setup: plot range
    x_bounds = [0,0]
    
    # loop through list of data
    for idx, data in enumerate(data_list):
    
        # 1) Data stuff
        # ------------------------------------------------
        type_key = data[0]
        name = data[2] #TODO: "name" is currently unused in ROOT plots, need to add a legend
        data = data[1] # TODO: careful, re-using namespace here!
        # Determine whether or not we're plotting percentage
        percentage = False
        if('(%)' in key_conversion[key][1]): percentage = True
        mult = 1.
        if(percentage): mult = 100.
        # remove any missing entries
        for dkey in list(data.keys()):
            if(data[dkey] == 'MISSING'): data.pop(dkey)
        if(data == {}): return
        x_vals = np.array(list(data.keys()),dtype=np.dtype('f8'))
        y_vals = mult * np.array([val[key_conversion[key][2]] for val in list(data.values())],dtype=np.dtype('f8'))
        n = len(y_vals)
        if(x_vals[-1]+1 > x_bounds[1]): x_bounds[1] = int(x_vals[-1]+1)
        # ------------------------------------------------

        #2) mpl
        ax.errorbar(x_vals,y_vals, linewidth=1.0, color = colors[idx%len(colors)], label = name)
        ax.set_xbound(lower=x_bounds[0],upper=x_bounds[1])
        ax.set_ybound(lower=axis_range[key][0],upper=axis_range[key][1])
        
        #3) ROOT
        if(use_ROOT):
            acc = rt.TGraph(n,x_vals,y_vals)
            axis = acc.GetXaxis()
            axis.Set(n+2,0,n+1)
            axis.SetNdivisions(n+1)
            root_title = 'ZZZ XXX vs. Epoch;Epoch;YYY'
            root_title = root_title.replace('XXX',key_conversion[key][0]).replace('YYY',key_conversion[key][1]).replace('ZZZ',type_conversion[type_key])
            rcolor = rt.TColor.GetColor(colors_hex[idx%len(colors_hex)])
            acc.SetLineColor(rcolor)
            acc.SetMarkerColor(rcolor)
            acc.SetTitle(root_title)
            canv.cd()
            if(idx == 0):
                acc.Draw()
                axes = [acc.GetXaxis(),acc.GetYaxis()]
                axes[0].Set(int(x_vals[-1]+2),0,int(x_vals[-1]+1))
                axes[0].SetNdivisions(n+1,False)
                axes[1].SetNdivisions(int(axis_range[key][2]),False) # how many divisions to draw (regardless of axis scaling)
                axes[1].SetRangeUser(axis_range[key][0],axis_range[key][1]) # the visible range (i.e. what's shown)
            else: acc.Draw('SAME') # will be drawn on same axis

        
    #4) mpl post
    plt.grid(b=True, which='major', color='0.65', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel(key_conversion[key][1])
    plt.title(title + ': ' + type_key)
    # adjust number of columns in the legend, based on number of entries
    ndata = len(data_list)
    ncol = int(np.ceil(ndata/6))
    if(key in ['acc','rej']):
        plt.legend(loc='upper left', prop={'size': 12}, ncol=ncol)
    elif(key in ['loss']):
        plt.legend(loc='lower left', prop={'size': 12}, ncol=ncol)
    else:
        plt.legend(loc='lower right', prop={'size': 12}, ncol=ncol)
    fig.tight_layout() # get rid of excess margins in image
    
    #5) ROOT post
    if(use_ROOT):
        canv.SetGrid()
        rt.gStyle.SetGridStyle(3)
        canv.Draw()
    
    #6) save plots
    filetypes = ['png','eps','pdf']
    savedir_full = 'summary/plt'
    if(savedir != ''): savedir_full = savedir + '/' + savedir_full
    savename = key + '_' + type_key
    for filetype in filetypes:
        dir = savedir_full + '/' + filetype
        try:os.makedirs(dir)
        except: pass
        full_savename = dir + '/' + savename + '_plt' + '.' + filetype
        fig.savefig(full_savename,transparent=False,dpi=300)
    plt.close()
    
    if(use_ROOT):
        filetypes = ['png','eps','pdf']
        savedir_full = 'summary/root'
        if(savedir != ''): savedir_full = savedir + '/' + savedir_full
        savename = key + '_' + type_key
        for filetype in filetypes:
            dir = savedir_full + '/' + filetype
            try:os.makedirs(dir)
            except: pass
            full_savename = dir + '/' + savename + '_root' + '.' + filetype
            canv.SaveAs(full_savename)
    
    # 7) Save the latest stats in a separate text file
    stats_file = type_key + '_avg.txt'
    if(savedir != ''): stats_file = savedir + '/' + stats_file
    val = y_vals[-1]
    write_type = 'a'
    if(overwrite): write_type = 'w'
    with open(stats_file,write_type) as f:
        print(f'{key}\t{val}',file=f)
    return
  
# Make individual plots for the average {loss, ACC, AUC, rejection} over averaged data (i.e. containing error bars)
def PlotAvg(avg_data, key, savedir = '', overwrite = False):
    type_key = avg_data[0]
    data = avg_data[1]
    nepochs = data[2]
    
    # Determine whether or not we're plotting percentage
    percentage = False
    if('(%)' in key_conversion[key][1]): percentage = True
    mult = 1.
    if(percentage): mult = 100.
    if(data == {}): return
    x_vals = np.array(list(data.keys()),dtype=np.dtype('f8'))
    n = len(x_vals)
    x_errs = np.zeros(n)
    y_vals = np.zeros(n,dtype=np.dtype('f8'))
    y_errs = np.zeros(n,dtype=np.dtype('f8'))
    for idx, val in enumerate(list(data.values())):
        y_vals[idx] = mult * val[0][key_conversion[key][2]]
        y_errs[idx] = mult * val[1][key_conversion[key][2]] / 2. # divide by 2 since we want this to be the distance between upper and lower error bars
    
    # 1) Make ROOT plot
    if(use_ROOT):
        acc = rt.TGraphErrors(n,x_vals,y_vals,x_errs,y_errs)
        title = 'Avg. ZZZ XXX vs. Epoch;Epoch;YYY'
        title = title.replace('XXX',key_conversion[key][0])
        title = title.replace('YYY',key_conversion[key][1])
        title = title.replace('ZZZ',type_conversion[type_key])
        acc.SetTitle(title)
        rt.gStyle.SetOptStat(0)
        canv = rt.TCanvas('c1','c1',800,600)
        canv.SetGrid()
        rt.gStyle.SetGridStyle(3)
        acc.Draw()
        # Adjust the axes
        x_axis = acc.GetXaxis()
        x_axis.Set(int(x_vals[-1]+2),0,int(x_vals[-1]+1))
        x_axis.SetNdivisions(n+1,False)
        y_axis = acc.GetYaxis()
        y_axis.SetNdivisions(int(axis_range[key][2]),False) # how many divisions to draw (regardless of axis scaling)
        y_axis.SetRangeUser(axis_range[key][0],axis_range[key][1]) # the visible range (i.e. what's shown)
        canv.Draw()
        filetypes = ['root','png','eps','pdf'] # PNG, PDF & EPS taken care of by visualize.C
        savename = key + '_' + type_key
        savedir_full = 'summary/root'
        if(savedir != ''): savedir_full = savedir + '/' + savedir_full
        for filetype in filetypes:
            dir = savedir_full + '/' + filetype
            try:os.makedirs(dir)
            except: pass
            full_savename = dir + '/' + savename + '_root' + '.' + filetype
            canv.SaveAs(full_savename)
            # include the TGraphErrors object in the ROOT file
            if(filetype == 'root'):
                rfile = rt.TFile(full_savename,'UPDATE')
                acc.Write('g1')
                rfile.Close()
    
    # 2) Make matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111) # why do we need to do these things in matplotlib?
    ax.errorbar(x_vals,y_vals,yerr=y_errs, linewidth=1.0, color = (0.,0.,0.))
    ax.set_xbound(lower=0,upper=int(x_vals[-1]+1))
    ax.set_ybound(lower=axis_range[key][0],upper=axis_range[key][1])
#    #percentage formatting where applicable
#    if('(%)' in key_conversion[key][1]): ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter())
    plt.grid(b=True, which='major', color='0.65', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel(key_conversion[key][1])
    fig.tight_layout() # get rid of excess margins in image
    filetypes = ['png','eps','pdf']
    savedir_full = 'summary/plt'
    if(savedir != ''): savedir_full = savedir + '/' + savedir_full
    for filetype in filetypes:
        dir = savedir_full + '/' + filetype
        try:os.makedirs(dir)
        except: pass
        full_savename = dir + '/' + savename + '_plt' + '.' + filetype
        fig.savefig(full_savename,transparent=False,dpi=300)
    plt.close()
    
    # 3) Save the latest stats in a separate text file
    stats_file = type_key + '_avg.txt'
    if(savedir != ''): stats_file = savedir + '/' + stats_file
    val = y_vals[-1]
    write_type = 'a'
    if(overwrite): write_type = 'w'
    with open(stats_file,write_type) as f:
        print(f'{key}\t{val}',file=f)
    return
      
# plot training and validation together on the same plots, as 2 data series (only with matplotlib)
def PlotAvgCombine(avg_data, key, savedir = ''):
    dlen = len(avg_data)
    type_key = [avg_data[i][0] for i in range(dlen)]
    data = [avg_data[i][1] for i in range(dlen)]
    nepochs = [avg_data[i][2] for i in range(dlen)]
    # Determine whether or not we're plotting percentage
    percentage = False
    if('(%)' in key_conversion[key][1]): percentage = True
    mult = 1.
    if(percentage): mult = 100.
    x_vals = [np.array(list(data[i].keys()),dtype=np.dtype('f8')) for i in range(dlen)]
    n = [len(x_vals[i]) for i in range(dlen)]
    x_errs = [np.zeros(n)] * dlen
    y_vals = [np.zeros(n[i],dtype=np.dtype('f8')) for i in range(dlen)]
    y_errs = [np.zeros(n[i],dtype=np.dtype('f8')) for i in range(dlen)]
    for i in range(dlen):
        for idx, val in enumerate(list(data[i].values())):
            y_vals[i][idx] = mult * val[0][key_conversion[key][2]]
            y_errs[i][idx] = mult * val[1][key_conversion[key][2]] / 2. # divide by 2 since we want this to be the distance between upper and lower error bars
    # Make matplotlib plot
    markersize = 6
    fontsize = 14
    labelsize = 14
    markers = ['o','s','^','*']
    markers = [None]
    linestyles = ['-','--','-.',':']
    colors = ['xkcd:kelly green','xkcd:true blue','xkcd:medium purple','xkcd:macaroni and cheese']
    fig = plt.figure()
    # make multiple plots -- expecting 2, for training and validation but we'll be flexible
    for i in range(dlen):
        ax = fig.add_subplot(111) # why do we need to do these things in matplotlib?
        ax.errorbar(x_vals[i],y_vals[i],yerr=y_errs[i], color = colors[i%len(colors)],marker = markers[i%len(markers)], linestyle = linestyles[i%len(linestyles)], markersize = markersize, label=type_conversion[type_key[i]], capsize = 3, markeredgewidth=1)
#        ax.errorbar(x_vals[i],y_vals[i],yerr=y_errs[i], color = colors[i%4],marker = markers[i%4], markersize = 5, linestyle = None, fmt = '', label=type_conversion[type_key[i]]) # need to pass fmt due to a stupid bug in matplotlib, otherwise lines will be drawn irrespective of linestyle var
        ax.tick_params('x',labelsize=labelsize)
        ax.tick_params('y',labelsize=labelsize)
        ax.set_xbound(lower=0,upper=int(x_vals[i][-1]+1))
        ax.set_ybound(lower=axis_range[key][0],upper=axis_range[key][1])
    plt.grid(b=True, which='major', color='0.65', linestyle='--')
    plt.xlabel('Epoch',fontsize=fontsize)
    plt.ylabel(key_conversion[key][1],fontsize=fontsize)
    plt.legend(loc='upper left', prop={'size': 12})
    fig.tight_layout() # get rid of excess margins in image
    filetypes = ['png','eps','pdf']
    savename = key + '_comb'
    savedir_full = 'summary/plt'
    if(savedir != ''): savedir_full = savedir + '/' + savedir_full
    for filetype in filetypes:
        dir = savedir_full + '/' + filetype
        try:os.makedirs(dir)
        except: pass
        full_savename = dir + '/' + savename + '_plt' + '.' + filetype
        fig.savefig(full_savename,transparent=False,dpi=300)
    plt.close()
    return
 
def Draw(run_folders, nepochs = 10, type_key = 'train',savedir = '', average = True, title = ''):
    if(average):
        data = GetAvgData(run_folders, nepochs, type_key)
        PlotAvg(data,'loss',savedir, True) # Loss
        PlotAvg(data,'acc',savedir) # Accuracy
        PlotAvg(data,'auc',savedir) # AUC
        PlotAvg(data,'rej',savedir) # Background rejection @ 0.3
    else:
        data = [GetData(run, nepochs, type_key) for run in sorted_nicely(run_folders)]
        Plot(data,'loss',savedir, title, True) # Loss
        Plot(data,'acc',savedir, title) # Accuracy
        Plot(data,'auc',savedir, title) # AUC
        Plot(data,'rej',savedir, title) # Background rejection @ 0.3
    return
   
def DrawOverlay(run_folders, nepochs = 10, type_keys = ['train','valid'],savedir = ''):
    data = [GetAvgData(run_folders,nepochs,key) for key in type_keys]
    for val in list(key_conversion.keys()):
        PlotAvgCombine(data,val,savedir)
    return

def DrawAll(run_folders, nepochs = 10, type_keys = ['train','valid'],savedir=''):
    data = [GetAvgData(run_folders,nepochs,key) for key in type_keys]
    # plots with training & validation series together
    for val in list(key_conversion.keys()):
        PlotAvgCombine(data,val,savedir)
    # separate plots for the two series
    for entry in data:
        for idx,val in enumerate(list(key_conversion.keys())):
            PlotAvg(entry,val,savedir, (idx==0))
    return
    

# TODO: The script currently assumes that the log files are stored in folders like /setN/ArunB/, where N is an integer, A & B are strings.
def main(args):
    # Set up some stuff for ROOT
    if(use_ROOT):
        rt.gROOT.ProcessLine('gErrorIgnoreLevel = kInfo+1;') # get rid of some printouts we don't really need
        rt.gROOT.SetBatch(True) # batch mode -> don't try to draw things on the screen
        rt.gROOT.SetStyle('ATLAS') # ATLAS Style built-in, should work for ROOT version > 6.13
        rt.gStyle.SetMarkerSize(0.6) # adjust marker size -- ATLAS style makes it rather large
        rt.gStyle.SetStripDecimals(False) # adjusting ATLAS style -- stripping decimals doesn't look neat to me
        rt.gStyle.SetOptStat(0) # we almost never want this stat box to appear
        
    set_folder = str(sys.argv[1])
    do_individual = int(sys.argv[2])
    do_avg = int(sys.argv[3])
    if(len(sys.argv) > 4): title = str(sys.argv[4])
    else: title = ''
    
    if(len(sys.argv) > 5): nepochs = int(sys.argv[5])
    else: nepochs = 10
    run_folders = [x for x in os.listdir(set_folder) if 'run' in x]
    run_folders = [set_folder + '/' + x for x in run_folders]
    
    types = []
    if(do_individual > 0):
        types += ['train','valid']
        Draw(run_folders,nepochs,'train',set_folder,False,title)
        Draw(run_folders,nepochs,'valid',set_folder,False,title)
    
    if(do_avg > 0):
        types += ['comb']
        DrawAll(run_folders,nepochs,['train','valid'],set_folder)
        #TODO: Some hard-coding for combining png plots using imagemagick
    summary_dir = set_folder + '/summary/plt/png'
    try:
        for type in types:
            sub.check_call(['convert','+append',summary_dir + '/acc_' + type + '_plt.png', summary_dir + '/auc_' + type + '_plt.png', summary_dir + '/a.png'])
            sub.check_call(['convert','+append',summary_dir + '/loss_' + type + '_plt.png', summary_dir + '/rej_' + type + '_plt.png', summary_dir + '/b.png'])
            sub.check_call(['convert','-append',summary_dir + '/a.png',summary_dir + '/b.png',summary_dir +'/summary_' + type + '_plt.png'])
            sub.check_call(['rm',summary_dir + '/a.png'])
            sub.check_call(['rm',summary_dir + '/b.png'])
    except:
        print('Couldn\'t concatenate matplotlib png files using imagemagicks (you might not have this tool installed).')
        pass
    return

if __name__ == '__main__':
    main(sys.argv)
    
## Make individual plots for the average {loss, ACC, AUC, rejection} for a single run (only with ROOT)
#def Plot(data, key, savedir = '', overwrite = False):
#    type_key = data[0]
#    data = data[1]
#    # Determine whether or not we're plotting percentage
#    percentage = False
#    if('(%)' in key_conversion[key][1]): percentage = True
#    mult = 1.
#    if(percentage): mult = 100.
#
#    # remove any missing entries
#    for dkey in list(data.keys()):
#        if(data[dkey] == 'MISSING'): data.pop(dkey)
#    if(data == {}): return
#    x_vals = np.array(list(data.keys()),dtype=np.dtype('f8'))
#    y_vals = mult * np.array([val[key_conversion[key][2]] for val in list(data.values())],dtype=np.dtype('f8'))
#    n = len(y_vals)
#    acc = rt.TGraph(n,x_vals,y_vals)
#    axis = acc.GetXaxis()
#    axis.Set(n+2,0,n+1)
#    axis.SetNdivisions(n+1)
#    title = 'ZZZ XXX vs. Epoch;Epoch;YYY'
#    title = title.replace('XXX',key_conversion[key][0])
#    title = title.replace('YYY',key_conversion[key][1])
#    title = title.replace('ZZZ',type_conversion[type_key])
#    # Some beautification beyond ATLAS style
#    acc.SetLineColor(rt.kBlue)
#    acc.SetMarkerColor(rt.kBlue)
#    acc.SetTitle(title)
#    rt.gStyle.SetOptStat(0)
#    canv = rt.TCanvas('c1','c1',800,600)
#    canv.SetGrid()
#    rt.gStyle.SetGridStyle(3)
#    acc.Draw()
#    # Adjust the axes
#    x_axis = acc.GetXaxis()
#    x_axis.Set(int(x_vals[-1]+2),0,int(x_vals[-1]+1))
#    x_axis.SetNdivisions(n+1,False)
#    y_axis = acc.GetYaxis()
#    y_axis.SetNdivisions(int(axis_range[key][2]),False) # how many divisions to draw (regardless of axis scaling)
#    y_axis.SetRangeUser(axis_range[key][0],axis_range[key][1]) # the visible range (i.e. what's shown)
#    canv.Draw()
#    savename = key + '_' + type_key + '.png'
#    if(savedir != ''): savename = savedir + '/' + savename
#    canv.SaveAs(savename)
#    # Also save the latest stats in a separate text file
#    stats_file = type_key + '.txt'
#    if(savedir != ''): stats_file = savedir + '/' + stats_file
#    val = y_vals[-1]
#    write_type = 'a'
#    if(overwrite): write_type = 'w'
#    with open(stats_file,write_type) as f:
#        print(f'{key}\t{val}',file=f)
#    return
  
