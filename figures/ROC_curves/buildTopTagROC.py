# Import numpy, pandas, HDF5
import numpy as np
import h5py
import pandas
import warnings
warnings.filterwarnings("ignore")
import sys, os, glob
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

# Function that takes the labels and score of the positive class
# (top class) and returns a ROC curve, as well as the signal efficiency
# and background rejection at a given targe signal efficiency, defaults
# to 0.3
def buildROC(labels, score, targetEff=0.3):
    fpr, tpr, threshold = roc_curve(labels, score)
    idx = np.argmin(np.abs(tpr - targetEff))
    eB, eS = fpr[idx], tpr[idx]
    return fpr, tpr, threshold, eB, eS

def main(args):
    fig = plt.figure(figsize=(7, 7))
    plt.rcParams.update({'font.size': 16})
    
    # CSV file with our network's ROC results
    # Must provide *full* absolute/relative path
    my_ROC_filename = str(sys.argv[1]) # e.g. ~/data/set7run1.valid_ROC.csv
    
    if(len(sys.argv) > 2): mode = str(sys.argv[2])
    else: mode = 'light'
    
    # Data directory for reference networks
    if(len(sys.argv) > 3): datadir_ref = str(sys.argv[3])
    else: datadir_ref = "./TopTagReference/"
    if (datadir_ref[-1] != '/'): datadir_ref = datadir_ref + '/'
    
    # Output directory
    if(len(sys.argv) > 4): outdir = str(sys.argv[4])
    else: outdir = "./output/"
    if (outdir[-1] != '/'): outdir = outdir + '/'
    try: os.makedirs(outdir) # like 'mkdir -p'
    except: pass

    # Output filename
    if(len(sys.argv) > 5): outfilename = str(sys.argv[5])
    else: outfilename = "ROCcomparisons"
    outfilename += '_' + mode

    # Prep the axis colors (silly matplotlib way of doing things...)
    theme_color = ['xkcd:black', '0.65','xkcd:white'] # text, grid line (grey scale)
    if(mode == 'dark'): theme_color = ['xkcd:white', '0.65','xkcd:black']
    mpl.rc('axes',edgecolor=theme_color[0],labelcolor = theme_color[0], facecolor = theme_color[0])
    mpl.rc('xtick',color=theme_color[0])
    mpl.rc('ytick',color=theme_color[0])

    # Test dataset and associated labels
    labels = []

    # If we want to read the original data file
    if(len(sys.argv) > 6): doFullTestFile= int(sys.argv[6])
    else: doFullTestFile = 1
    doFullTestFile = (doFullTestFile == 1)

    if doFullTestFile:
        if(len(sys.argv) > 7): testfilepath = str(sys.argv[7])
        else: testfilepath = "/Users/jan/git_repos/njet/NBodyJetNets/DataSamples/toptag/data/raw/test.h5"
    
        testfile_exists = (len(glob.glob(testfilepath)) == 1)
        if(not testfile_exists):
            print('Uh oh, doFullTestFile = True but file ' + testfilepath + ' not found. Exiting.')
            return
        testdata     = pandas.HDFStore(testfilepath)

        # Labels from test dataset
        labels       = testdata.select("table",columns=["is_signal_new"])
        # Otherwise, we must have saved the labels to a dedicated file
#    else:
#        testlabelsfilename = "testlabels.npy" # TODO: This file does not exist in the repo! Did it ever?
#        labels = np.load(datadir_ref + testlabelsfilename)

        # List of results to load. These come from the Top Tagging Reference Dataset
        # which is described here: https://docs.google.com/document/d/1Hcuc6LBxZNX16zjEGeq16DAzspkDC4nDTyjMp1bWHRo/edit
        # and hosted here: https://desycloud.desy.de/index.php/s/4DDFkfYRGaPo2WJ
        results =	{
        #  "NSub(6)": "6_body_test_sets.npy",
        #  "DGCNN": "DGCNN_v1.npy",
        "EFN": "EFN_v0.npy",
        "EFP": "EFP_v0.npy",
        #  "LDA": "LDA_testScores.npy",
        "P-CNN": "P-CNN_v2.npy",
        "PFN": "PFN_v0.npy",
        "ParticleNet": "ParticleNet_v2.npy",
        "ResNeXt50": "ResNeXt50_v2.npy",
        #  "R-CNN": "RutgersCNN_9runs.npy",
        #  "TreeNiN": "TreeNiN_hd50.npy",
        #  "LBN": "lbn_ensemble9.npz",
        #  "NSub(8)": "nsub_8body_v2.npy",
        "TopoDNN": "topodnn_v2.npy",
        }

        # Loop over the results and extract the score for the signal class
        # then build a ROC curve
    
        lines=iter(['dotted', 'dashed', 'dashdot', (0, (1, 1)),  (0, (3, 5, 1, 5)), (0, (3, 1, 1, 1)), (0, (3, 5, 1, 5, 1, 5))])
        for network in results:
            print("Building ROC curve for network: %s" % network)
            filename = results[network]  # Get file name
            score    = np.load(datadir_ref + filename)   # Load file and get scores
            fpr, tpr, threshold, eB, eS = buildROC(labels, score, targetEff=0.3)
            rejection = 1./eB
            #plt.plot(tpr, 1/fpr, label = r'%s ($1/\epsilon_{B}$ = %0.2f)' % (network, rejection) )
            plt.plot(tpr, 1/fpr, label = '%s' % (network), linestyle = next(lines))


    #Our results
    my_roc    = np.loadtxt(my_ROC_filename, delimiter=',')
    my_fpr    = my_roc[0]
    my_tpr    = my_roc[1]
    my_thresh = my_roc[2]
    #plt.plot(my_tpr, 1/my_fpr, label = r'%s ($1/\epsilon_{B}$ = %0.2f)' % ('CLARIANT', 426), linewidth=3.0, color='#4bade9')
    plt.plot(my_tpr, 1/my_fpr, label = '%s' % ('LGN'), linewidth=1., color='#4bade9')

    # method I: plt
    #plt.title('Receiver Operating Characteristic')
    legend = plt.legend(loc = 'upper right',prop={'size':15})
    plt.setp(legend.get_texts(), color = theme_color[0])
    legend.get_frame().set_facecolor(theme_color[2])
    plt.plot([0, 1], [0, 1],'r--')
    
    plt.xlim([0, 1])
    plt.ylim([4, 40000])
    plt.yscale('log')
    
    plt.grid(True)
    plt.grid(b=True, which='both', color=theme_color[1], linestyle='--')
    
    plt.xticks(np.arange(0, 1, step=0.1))

    plt.ylabel(r'Background rejection $\frac{1}{\epsilon_{B}}$', color = theme_color[0])
    plt.xlabel(r'Signal efficiency $\epsilon_{S}$', color = theme_color[0])

    plt.savefig(outdir + outfilename + ".png", transparent=True, dpi=300)
    plt.savefig(outdir + outfilename + ".svg", transparent=True, dpi=300, format='svg')
    plt.show()

if __name__ == '__main__':
    main(sys.argv)
