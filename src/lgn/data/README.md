Dec 31, 2019: 
Moved the existing `utils.py` -> `utils_old.py` .

The new `utils.py` should allow for the concatenation of multiple HDF5 files into a single dataset object, via `torch.utils.data.ConcatDataset`.
