Additional utilities, outside of plotting & file format conversion.

`pt_sort.py`: Sorts the events in the HDF5 format files, by jet pT. (Files must contain the "jet_pt" column, added March 2020).

`pt_slice.py`: Create copies of the HDF5 format files, containing events that fall within a user-specified jet pT window.

`slice.py`: Automatically run `pt_slice.py` over a range of jet pT windows. Note: Arguments are *hard-coded*.

`beam_adjust.py`: Allows for the adjustment of the masses & labels of beam particles in the HDF5 format files, if they have been added via the appropriate options in the `raw2h5` conversion tools. (DEPRECATED)
