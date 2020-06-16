import sys
import ROOT as rt

# Take the reduced ROOT file,
# and split it into 3 based
# on the values of the
# ttv branch in the TTree.
def split_ttv(input_filename):
    input_file = rt.TFile(input_filename,'READ')
    input_tree = input_file.Get('events_reduced')
    branch_names = [x.GetName() for x in input_tree.GetListOfBranches()]
    if('ttv' not in branch_names):
        print('Error: Did not find branch \'ttv\' in',input_filename + '.')
        return
    output_names = [input_filename.replace('.root','_'+x+'.root') for x in ['train','test','valid']]
    output_files = [rt.TFile(x,'RECREATE') for x in output_names]
    for i in range(len(output_files)):
        output_files[i].cd()
        t = input_tree.CopyTree('ttv=='+str(i))
        t.Write('',rt.TObject.kOverwrite)
    input_file.Close() # must close *before* the output_files, otherwise we have a crash (virtual function called)
    for i in range(len(output_files)): output_files[i].Close()

def main(args):
    input_file = str(sys.argv[1])
    split_ttv(input_file)
    return

if __name__ == '__main__':
    main(sys.argv)

