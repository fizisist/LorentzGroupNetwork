Additional utilities, outside of plotting & file format conversion. All of these are to be run on the converted dataset, not the original `pandas` HDFStore file.

`pt_sort.py`: Create copies of the data files, with events sorted by jet pT. (Files must contain the "jet_pt" column, added March 2020).

`pt_slice.py`: Create copies of the  data files, only containing events that fall within a user-specified jet pT window.

`slice.py`: Automatically run `pt_slice.py` over a range of jet pT windows. Note: Arguments are *hard-coded*.

`reduce.py`: Create copies of the data files, that only contain some % of the original events (while preserving the signal/background ratio).

`rescale.py`: Given two data files, remove events from the 1st file so that it has the same number of signal and background events as the 2nd. This is useful after running `pt_slice.py`, as different pT slices may have different numbers of events due to a non-uniform pT distribution.

`beam_adjust.py`: Allows for the adjustment of the masses & labels of beam particles in the HDF5 format files, if they have been added via the appropriate options in the `raw2h5` conversion tools. (DEPRECATED)
