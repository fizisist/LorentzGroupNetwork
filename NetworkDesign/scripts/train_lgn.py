import torch
from torch.utils.data import DataLoader

import logging
from datetime import datetime
from math import sqrt

from lgn.models import LGNCG, LGNTopTag
from lgn.models.autotest import lgn_tests

from lgn.engine import Trainer
from lgn.engine import init_argparse, init_file_paths, init_logger, init_cuda, logging_printout, fix_args
from lgn.engine import init_optimizer, init_scheduler
from lgn.data.utils import initialize_datasets
from lgn.cg_lib import CGDict

from lgn.data.collate import collate_fn

# This makes printing tensors more readable.
torch.set_printoptions(linewidth=1000, threshold=100000)

logger = logging.getLogger('')


def main():

    # Initialize arguments -- Just
    args = init_argparse()
   
    # Initialize file paths
    args = init_file_paths(args)

    # Initialize logger
    init_logger(args)

    # Write input paramaters and paths to log
    logging_printout(args)

    # Fix possible inconsistencies in arguments
    args = fix_args(args)

    # Initialize device and data type
    device, dtype = init_cuda(args)

    # Initialize dataloder
    args, datasets = initialize_datasets(args, args.datadir, num_pts=None)

    if args.task.startswith('eval'):
        args.load = True
        args.num_epoch = 1

    # Construct PyTorch dataloaders from datasets
    collate = lambda data: collate_fn(data, scale=args.scale, nobj=args.nobj, add_beams=args.add_beams, beam_mass=args.beam_mass)
    dataloaders = {split: DataLoader(dataset,
                                     batch_size=args.batch_size,
                                     shuffle=args.shuffle if (split == 'train') else False,
                                     num_workers=args.num_workers,
                                     collate_fn=collate)
                   for split, dataset in datasets.items()}

    # Initialize model
    model = LGNTopTag(args.maxdim, args.max_zf, args.num_cg_levels, args.num_channels,
                      args.cutoff_type, args.hard_cut_rad, args.soft_cut_rad, args.soft_cut_width,
                      args.weight_init, args.level_gain, args.num_basis_fn,
                      args.top, args.input, args.num_mpnn_levels, activation=args.activation, pmu_in=args.pmu_in, add_beams=args.add_beams,
                      mlp=args.mlp, mlp_depth=args.mlp_depth, mlp_width=args.mlp_width,
                      scale=1., full_scalars=args.full_scalars,
                      device=device, dtype=dtype)
    
    if args.parallel:
        model = torch.nn.DataParallel(model)

    # Initialize the scheduler and optimizer
    optimizer = init_optimizer(args, model)
    scheduler, restart_epochs = init_scheduler(args, optimizer)

    # Define a loss function.
    # loss_fn = torch.nn.functional.cross_entropy
    loss_fn = torch.nn.CrossEntropyLoss().cuda()
    
    # Apply the covariance and permutation invariance tests.
    if args.test:
        lgn_tests(model, dataloaders['train'], args, cg_dict=model.cg_dict)

    # Instantiate the training class
    trainer = Trainer(args, dataloaders, model, loss_fn, optimizer, scheduler, restart_epochs, device, dtype)
    
    # Load from checkpoint file. If no checkpoint file exists, automatically does nothing.
    trainer.load_checkpoint()

    # Train model.
    trainer.train()

    # Test predictions on best model and also last checkpointed model.
    trainer.evaluate(splits=['test'])

if __name__ == '__main__':
    main()
