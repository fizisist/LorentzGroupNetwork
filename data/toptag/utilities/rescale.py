# rescale.py
# Created by Jan Offermann on 09/22/20.
#
# Goal: Given two data files, rescale the
#       first as to have the same # of signal
#       and background events as the second.
#       Preserves existing event ordering.

import sys
import h5py as h5, numpy as np

def main(args):
    input_file = sys.argv[1] # file of which to make a rescaled copy
    model_file = sys.argv[2] # file to match for # of signal & background events
    rescale(input_file, model_file)
    return

def rescale(input, model):
    f_input = h5.File(input,'r')
    f_model = h5.File(model, 'r')
    
    # Get the number of s/b events for each
    n_input = np.array([np.sum(f_input['is_signal'][:]),f_input['is_signal'].shape[0]],dtype=np.dtype('i8'))
    n_model = np.array([np.sum(f_model['is_signal'][:]),f_model['is_signal'].shape[0]],dtype=np.dtype('i8'))
    
    # At this point we can safely close f_model
    f_model.close()
    
    size_check = np.greater(n_input,n_model)
    if(size_check[0] == False):
        print('Error: Model file has more signal events than input file => cannot rescale input.')
        return
    if(size_check[1] == False):
        print('Error: Model file has more background events than input file => cannot rescale input.')
        return
      
    print('Rescaling # of signal events    : '+str(n_input[0])+' => '+str(n_model[0]) + '.')
    print('Rescaling # of background events: '+str(n_input[1])+' => '+str(n_model[1]) + '.')

    # Determine how many signal and background events to remove from the input file.
    n_remove = n_input - n_model
    
    # Get the lists of indices corresponding with signal and background files.
    # These are in increasing order (i.e. ordering is preserved in each).
    idxs = [np.flatnonzero(f_input['is_signal'][:] == 1),np.flatnonzero(f_input['is_signal'][:] == 0)]

    # Now randomly remove some indices from each list.
    idxs = [np.setdiff1d(idxs[x],np.random.choice(idxs[x],n_remove[x],replace=False)) for x in range(2)]
    
    # Now join the lists of signal and background indices, and sort to preserve original order.
    idxs = np.sort(np.concatenate((idxs[0],idxs[1])))

    # Now copy the input file, using only the index list we've gathered.
    copy = input.replace('.h5','_rescaled.h5')
    f_copy = h5.File(copy,'w')
    keys = list(f_input.keys())
    
    # Note: h5py seemingly doesn't let us retrieve data using non-sequential arrays as indices,
    # so we can't do a fancy one-liner, i.e. f_input[key][idxs] will throw a TypeError.
    # Thus we will load all the data into memory and put it in a numpy array.
    f_input_data = {key: f_input[key][:] for key in keys}
    f_copy_data = {key: f_input_data[key][idxs] for key in keys}
    
    for key in keys:
        print(key)
        f_copy.create_dataset(key,data=f_copy_data[key],compression = 'gzip')
    # Now close the remaining files.
    f_copy.close()
    f_input.close()
    return
    
if __name__ == '__main__':
    main(sys.argv)
