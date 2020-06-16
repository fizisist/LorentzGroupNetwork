# Training Datasets

## jet
Each entry contains a list of the `n` highest-`pT` constituents of the highest-`pT` jet from either a top pair production event (signal) or a QCD dijet event (background). By default, `n` = 200.

## toptag
This directory holds scripts used to convert the "Top-Tagging Reference Dataset", which can be found here: https://desycloud.desy.de/index.php/s/llbX3zpLhazgPJ6, to a format that our network's DataLoader can use.
The format of the converted files is identical to the `jet` datasets: Each entry contains a list of the 200 highest-`pT` constituents of the highest-`pT` jet from either a top pair production event (signal) or a QCD dijet event (background).
