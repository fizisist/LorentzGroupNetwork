`perf_plot.py`: Creates plots of network metrics (accuracy, area under the ROC curve, loss, background rejection @ 30% signal efficiency) as a function of the training epoch.

`build_roc_curves.py`: Creates ROC curves for batches of trained networks. Assumes that the first argument is a directory, containing subdirectories that each contain the .csv and .pt files from individual training sessions.
