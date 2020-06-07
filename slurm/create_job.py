#!/usr/bin/env python
import os, sys

prefix = 'jan3'
datadir = './data/samples_h5/jet/run12/parton/noMPI'
maxdim = 3
max_zf = 1
target = 'is_signal'
num_workers=0
num_channelss=[3]
level_gain = 1
num_epoch = 10
batch_size = 10
num_basis_fn = 10
num_cg_levels = 3
alpha = 50
scale = 0.01
lr_init = 0.001
lr_final = 0.00001
full_scalars = False
activation = 'elu'
mlp = True

jobs = []
for num_channels in num_channelss:
	targetstring = f"python3 scripts/train_lgn.py --prefix={prefix} --datadir={datadir} --maxdim={maxdim} --max-zf={max_zf} --target={target} --num-workers={num_workers} --num-channels={num_channels} --level-gain={level_gain} --num-epoch={num_epoch} --batch-size={batch_size} --num-basis-fn={num_basis_fn} --num-cg-levels={num_cg_levels} --alpha={alpha} --scale={scale} --lr-init={lr_init} --lr-final={lr_final} --full-scalars={full_scalars} --activation={activation} --mlp={mlp}\n"
	jobs.append(targetstring)
print(''.join(jobs))
