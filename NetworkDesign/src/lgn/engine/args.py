import argparse

from math import inf

#### Argument parser ####

def setup_argparse():

    parser = argparse.ArgumentParser(description='LGN network options')

    parser.add_argument('--task', type=str, default='train', metavar='str',
                        help='Train or evaluate model. (train | eval)')

    # Optimizer options
    parser.add_argument('--num-epoch', type=int, default=6, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--batch-size', '-bs', type=int, default=10, metavar='N',
                        help='Mini-batch size (default: 10)')
    parser.add_argument('--batch-group-size', '-bgs', type=int, default=1, metavar='N',
                        help='Mini-batch size (default: 10)')    

    parser.add_argument('--weight-decay', type=float, default=0, metavar='N',
                        help='Set the weight decay used in optimizer (default: 0)')
    parser.add_argument('--cutoff-decay', type=float, default=0, metavar='N',
                        help='Set the weight decay used in optimizer for learnable radial cutoffs (default: 0)')
    parser.add_argument('--lr-init', type=float, default=0.001, metavar='N',
                        help='Initial learning rate (default: 0.005)')
    parser.add_argument('--lr-final', type=float, default=1e-5, metavar='N',
                        help='Final (held) learning rate (default: 1e-5)')
    parser.add_argument('--lr-decay', type=int, default=inf, metavar='N',
                        help='Timescale over which to decay the learning rate (default: inf)')
    parser.add_argument('--lr-decay-type', type=str, default='cos', metavar='str',
                        help='Type of learning rate decay. (cos | linear | exponential | pow | restart) (default: cos)')
    parser.add_argument('--lr-minibatch', '--lr-mb', action=BoolArg, default=True,
                        help='Decay learning rate every minibatch instead of epoch.')
    parser.add_argument('--sgd-restart', type=int, default=-1, metavar='int',
                        help='Restart SGD optimizer every (lr_decay)^p epochs, where p=sgd_restart. (-1 to disable) (default: -1)')

    parser.add_argument('--optim', type=str, default='amsgrad', metavar='str',
                        help='Set optimizer. (SGD, AMSgrad, Adam, RMSprop)')
    parser.add_argument('--parallel', action=BoolArg, default=False,
                        help='Use nn.DataParallel when multiple GPUs are available.')

    # Dataloader and randomness options
    parser.add_argument('--shuffle', action=BoolArg, default=True,
                        help='Shuffle minibatches.')
    parser.add_argument('--seed', type=int, default=1, metavar='N',
                        help='Set random number seed. Set to -1 to set based upon clock.')

    # Saving and logging options
    parser.add_argument('--alpha', type=float, default=50, metavar='N',
                    help='Averaging exponent for recent loss printouts [0, inf), the higher the more smoothing (default = 50')
    parser.add_argument('--save', action=BoolArg, default=True,
                        help='Save checkpoint after each epoch. (default: True)')
    parser.add_argument('--load', action=BoolArg, default=False,
                        help='Load from previous checkpoint. (default: False)')

    parser.add_argument('--test', action=BoolArg, default=True,
                        help='Perform automated network testing. (Default: True)')

    parser.add_argument('--log-level', type=str, default='info',
                        help='Logging level to output')

    parser.add_argument('--textlog', action=BoolArg, default=True,
                        help='Log a summary of each mini-batch to a text file.')

    parser.add_argument('--predict', action=BoolArg, default=True,
                        help='Save predictions. (default: True)')

    parser.add_argument('--quiet', action=BoolArg, default=True,
                        help='Hide warnings about unused parameters. (default: True)')

    ### Arguments for files to save things to
    # Job prefix is used to name checkpoint/best file
    parser.add_argument('--prefix', '--jobname', type=str, default='nosave',
                        help='Prefix to set load, save, and logfile. (default: nosave)')

    # Allow to manually specify file to load
    parser.add_argument('--loadfile', type=str, default='',
                        help='Set checkpoint file to load. Leave empty to auto-generate from prefix. (default: (empty))')
    # Filename to save model checkpoint to
    parser.add_argument('--checkfile', type=str, default='',
                        help='Set checkpoint file to save checkpoints to. Leave empty to auto-generate from prefix. (default: (empty))')
    # Filename to best model checkpoint to
    parser.add_argument('--bestfile', type=str, default='',
                        help='Set checkpoint file to best model to. Leave empty to auto-generate from prefix. (default: (empty))')
    # Filename to save logging information to
    parser.add_argument('--logfile', type=str, default='',
                        help='Duplicate logging.info output to logfile. Set to empty string to generate from prefix. (default: (empty))')
    # Filename to save predictions to
    parser.add_argument('--predictfile', type=str, default='',
                        help='Save predictions to file. Set to empty string to generate from prefix. (default: (empty))')

    # Working directory to place all files
    parser.add_argument('--workdir', type=str, default='./',
                        help='Working directory as a default location for all files. (default: ./)')
    # Directory to place logging information
    parser.add_argument('--logdir', type=str, default='log/',
                        help='Directory to place log and savefiles. (default: log/)')
    # Directory to place saved models
    parser.add_argument('--modeldir', type=str, default='model/',
                        help='Directory to place log and savefiles. (default: model/)')
    # Directory to place model predictions
    parser.add_argument('--predictdir', type=str, default='predict/',
                        help='Directory to place log and savefiles. (default: predict/)')
    # Directory to read and save data from
    parser.add_argument('--datadir', type=str, default='data/',
                        help='Directory to look up data from. (default: data/)')

    # Dataset options
    parser.add_argument('--dataset', type=str, default='jet',
                        help='Data set. Options: (jet, 2v3bodycomplex). Default: jet.')
    parser.add_argument('--target', type=str, default='is_signal',
                        help='Learning target for a dataset (such as qm9) with multiple options.')
    
    parser.add_argument('--nobj', type=int, default=None, metavar='N',
                        help='Max number of particles in each event (selects the first nobj). Set to None to use entire dataset. (default: None)')
    parser.add_argument('--num-train', type=int, default=-1, metavar='N',
                        help='Number of samples to train on. Set to -1 to use entire dataset. (default: -1)')
    parser.add_argument('--num-valid', type=int, default=-1, metavar='N',
                        help='Number of validation samples to use. Set to -1 to use entire dataset. (default: -1)')
    parser.add_argument('--num-test', type=int, default=-1, metavar='N',
                        help='Number of test samples to use. Set to -1 to use entire dataset. (default: -1)')
    parser.add_argument('--add-beams', action=BoolArg, default=True,
                        help='Append two proton beams of the form (m^2,0,0,+-1) to each event')
    parser.add_argument('--beam-mass', type=float, default=1, metavar='N',
                    help='Set mass m of the beams, so that E=sqrt(1 + m^2) (default = 1)')
    parser.add_argument('--force-download', action=BoolArg, default=False,
                        help='Force download and processing of dataset.')

    # Computation options
    parser.add_argument('--cuda', dest='cuda', action='store_true',
                        help='Use CUDA (default)')
    parser.add_argument('--no-cuda', '--cpu', dest='cuda', action='store_false',
                        help='Use CPU')
    parser.set_defaults(cuda=True)

    parser.add_argument('--float', dest='dtype', action='store_const', const='float',
                        help='Use floats.')
    parser.add_argument('--double', dest='dtype', action='store_const', const='double',
                        help='Use doubles.')
    parser.set_defaults(dtype='float')

    parser.add_argument('--num-workers', type=int, default=0,
                        help='Set number of workers in dataloader. (Default: 0)')

    # Model options
    parser.add_argument('--pmu-in', action=BoolArg, default=False,
                        help='Feed 4-momenta themselves to the first CG layer, in addition to scalars (default = False)')

    parser.add_argument('--num-cg-levels', type=int, default=3, metavar='N',
                        help='Number of CG levels (default: 3)')

    parser.add_argument('--mlp-depth', type=int, default=3, metavar='N',
                        help='Number of hidden layers in each MLP (default: 3)')
    parser.add_argument('--mlp-width', type=int, default=2, metavar='N',
                        help='Width of hidden layers in each MLP in units of the # of inputs (default: 2)')

    parser.add_argument('--maxdim', nargs='*', type=int, default=[3], metavar='N',
                        help='Cutoff in CG operations (default: [3])')
    parser.add_argument('--max-zf', nargs='*', type=int, default=[1], metavar='N',
                        help='Number of spherical harmonic powers to use (default: [1])')
    parser.add_argument('--num-channels', nargs='*', type=int, default=[2, 3, 4, 3], metavar='N',
                        help='Number of channels to allow after mixing (default: [3])')
    parser.add_argument('--level-gain', nargs='*', type=float, default=[1.], metavar='N',
                        help='Gain at each level (default: [1.])')

    parser.add_argument('--hard-cutoff', dest='hard_cut_rad',
                        type=float, default=inf, nargs='*', metavar='N',
                        help='Radius of HARD cutoff in Angstroms (default: inf)')
    parser.add_argument('--soft-cutoff', dest='soft_cut_rad', type=float,
                        default=inf, nargs='*', metavar='N',
                        help='Radius of SOFT cutoff in Angstroms (default: inf)')
    parser.add_argument('--soft-width', dest='soft_cut_width',
                        type=float, default=inf, nargs='*', metavar='N',
                        help='Width of SOFT cutoff in Angstroms (default: inf)')
    parser.add_argument('--cutoff-type', '--cutoff', type=str, default=['learn'], nargs='*', metavar='str',
                        help='Types of cutoffs to include')

    parser.add_argument('--num-basis-fn', type=int, default=10, metavar='N',
                        help='NUMBER OF BASIS FUNCTIONS (default: 10)')

    parser.add_argument('--scale', type=float, default=0.005, metavar='N',
                    help='Global scaling factor for input four-momenta (default = 0.001)')

    parser.add_argument('--full-scalars', action=BoolArg, default=False,
                    help='Feed the norms of ALL irrep tensors at each level into the output layer (default = False)')
   
    parser.add_argument('--mlp', action=BoolArg, default=True,
                    help='Insert a perceptron acting on invariant scalars inside each CG level (default = True)')
    parser.add_argument('--activation', type=str, default='leakyrelu',
                        help='Activation function used in MLP layers. Options: (relu, elu, leakyrelu, sigmoid, logsigmoid). Default: elu.')                


    # TODO: Update(?)
    parser.add_argument('--weight-init', type=str, default='randn', metavar='str',
                        help='Weight initialization function to use (default: rand)')

    parser.add_argument('--input', type=str, default='linear',
                        help='Function to apply to process l0 input (linear | MPNN) default: linear')
    parser.add_argument('--num-mpnn-levels', type=int, default=1,
                        help='Number levels to use in input featurization MPNN. (default: 1)')
    parser.add_argument('--top', '--output', type=str, default='linear',
                        help='Top function to use (linear | PMLP) default: linear')

    parser.add_argument('--gaussian-mask', action='store_true',
                        help='Use gaussian mask instead of sigmoid mask.')

    return parser

###

class BoolArg(argparse.Action):
    """
    Take an argparse argument that is either a boolean or a string and return a boolean.
    """
    def __init__(self, default=None, nargs=None, *args, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")

        # Set default
        if default is None:
            raise ValueError("Default must be set!")

        default = _arg_to_bool(default)

        super().__init__(*args, default=default, nargs='?', **kwargs)

    def __call__(self, parser, namespace, argstring, option_string):

        if argstring is not None:
            # If called with an argument, convert to bool
            argval = _arg_to_bool(argstring)
        else:
            # BoolArg will invert default option
            argval = True

        setattr(namespace, self.dest, argval)

def _arg_to_bool(arg):
    # Convert argument to boolean

    if type(arg) is bool:
        # If argument is bool, just return it
        return arg

    elif type(arg) is str:
        # If string, convert to true/false
        arg = arg.lower()
        if arg in ['true', 't', '1']:
            return True
        elif arg in ['false', 'f', '0']:
            return False
        else:
            return ValueError('Could not parse a True/False boolean')
    else:
        raise ValueError('Input must be boolean or string! {}'.format(type(arg)))


# From https://stackoverflow.com/questions/12116685/how-can-i-require-my-python-scripts-argument-to-be-a-float-between-0-0-1-0-usin
class Range(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
    def __eq__(self, other):
        return self.start <= other <= self.end
