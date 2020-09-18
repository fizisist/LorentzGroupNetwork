import torch
from lgn.g_lib import rotations as rot
import warnings

from itertools import zip_longest

from lgn.g_lib import g_tau, g_tensor
from lgn.g_lib import g_vec, g_scalar, g_weight, g_wigner_d
from lgn.g_lib.cplx_lib import mul_zscalar_zirrep, mul_zscalar_zscalar
from lgn.g_lib.cplx_lib import mix_zweight_zvec, mix_zweight_zscalar

GTau = g_tau.GTau
GTensor = g_tensor.GTensor
GVec = g_vec.GVec
GScalar = g_scalar.GScalar
GWeight = g_weight.GWeight
GWignerD = g_wigner_d.GWignerD


def _check_keys(val1, val2):
    if type(val1) in [list, tuple] or type(val2) in [list, tuple]:
        if len(val1) != len(val2):
            raise ValueError('Need as many scalars as irreps in the GTensor '
                             '({} {})!'.format(len(val1), len(val2)))
    elif val1.keys() != val2.keys():
        raise ValueError('Two GTensor subclasses contain different irreps '
                         '({} {})!'.format(val1.keys(), val2.keys()))


def _check_mult_compatible(val1, val2):
    """
    Function to check that two GTensors are compatible with regards
    to a specific binary operation.
    """
    
    if (val1.rdim is not None) and (val2.rdim is not None):
        # raise ValueError('Cannot multiply two GVecs together!')
        warnings.warn('Both Inputs have representation dimensions. '
                      'Multiplying them together may break covariance.',
                      RuntimeWarning)


def _dispatch_op(op, val1, val2):
    """
    Used to dispatch a binary operator where at least one of the two inputs is a
    GTensor.
    """

    # Hack to make GVec/GScalar multiplication work
    # TODO: Figure out better way of doing this?
    if isinstance(val1, GScalar) and isinstance(val2, GVec):
        _check_keys(val1, val2)
        applied_op = {key: op(val1[key].unsqueeze(val2.rdim), val2[key]) for key in val1.keys()}
        output_class = GVec
    elif isinstance(val1, GVec) and isinstance(val2, GScalar):
        _check_keys(val1, val2)
        applied_op = {key: op(val1[key], val2[key].unsqueeze(val1.rdim)) for key in val1.keys()}
        output_class = GVec
    elif isinstance(val1, GVec) and type(val2) in (int, float):
        applied_op = {key: op(val2, val) for key, val in val1.items()}
        output_class = GVec
    elif isinstance(val2, GVec) and type(val1) in (int, float):
        applied_op = {key: op(val1, val) for key, val in val2.items()}
    elif (isinstance(val1, GVec) or isinstance(val1, GTensor)) and type(val2) is dict:
        _check_keys(val1, val2)
        applied_op = {key: op(val1[key], val2[key]) for key in val1.keys()}
        output_class = type(val1)
    elif (isinstance(val2, GVec) or isinstance(val2, GTensor)) and type(val1) is dict:
        _check_keys(val1, val2)
        applied_op = {key: op(val1[key], val2[key]) for key in val2.keys()}
        output_class = type(val2)
    # Both va1 and val2 are other instances of GTensor
    elif isinstance(val1, GTensor) and isinstance(val2, GTensor):
        _check_keys(val1, val2)
        applied_op = {key: op(val1[key], val2[key]) for key in val1.keys()}
        output_class = type(val2)
    # Multiply val1 with a list/tuple
    elif isinstance(val1, GTensor) and type(val2) is dict:
        _check_keys(val1, val2)
        applied_op = {key: op(val1[key], val2[key]) for key in val1.keys()}
        output_class = type(val1)
    # Multiply val1 with something else
    elif isinstance(val1, GTensor) and not isinstance(val2, GTensor):
        applied_op = {key: op(val2, part1) for key, part1 in val1.items()}
        output_class = type(val1)
    # Multiply val2 with a list/tuple
    elif not isinstance(val1, GTensor) and type(val1) is dict:
        _check_keys(val1, val2)
        applied_op = {key: op(val1[key], val2[key]) for key in val1}
        output_class = type(val1)
    # Multiply val2 with something else
    elif not isinstance(val1, GTensor) and isinstance(val2, GTensor):
        applied_op = {key: op(val1, part2) for key, part2 in val2.items()}
        output_class = type(val2)
    else:
        raise ValueError('Neither class inherits from GTensor!')

    return output_class(applied_op)


def _dispatch_mul(val1, val2):
    """
    Used to dispatch a binary operator where at least one of the two inputs is a
    GTensor.
    """

    # Hack to make GVec/GScalar multiplication work
    # TODO: Figure out better way of doing this?
    if isinstance(val1, GScalar) and isinstance(val2, GVec):
        _check_keys(val1, val2)
        applied_op = {key: mul_zscalar_zirrep(val1[key], val2[key], rdim=val2.rdim)
                      for key in val1.keys()}
        output_class = GVec
    elif isinstance(val1, GVec) and isinstance(val2, GScalar):
        _check_keys(val1, val2)
        applied_op = {key: mul_zscalar_zirrep(val2[key], val1[key], rdim=val1.rdim)
                      for key in val1.keys()}
        output_class = GVec
    elif isinstance(val1, GScalar) and isinstance(val2, GScalar):
        _check_keys(val1, val2)
        applied_op = {key: mul_zscalar_zscalar(val1[key], val2[key])
                      for key in val1.keys()}
        output_class = GScalar
    # Both va1 and val2 are other instances of GTensor
    elif isinstance(val1, GTensor) and isinstance(val2, GTensor):
        _check_keys(val1, val2)
        _check_mult_compatible(val1, val2)
        applied_op = {key: mul_zscalar_zscalar(val1[key], val2[key])
                      for key in val1.keys()}
        output_class = type(val2)
    # Multiply val1 with a list/tuple
    elif not isinstance(val2, GTensor) and type(val2) in [list, tuple]:
        _check_keys(val1, val2)
        applied_op = [{key: torch.mul(scalar, part) for key, part in val1.items()} for scalar in val2]
        output_class = type(val1)
    # Multiply val1 with something else
    elif isinstance(val1, GTensor) and not isinstance(val2, GTensor):
        applied_op = {key: torch.mul(val2, val) for key, val in val1.items()}
        output_class = type(val1)
    # Multiply val2 with a list/tuple
    elif not isinstance(val1, GTensor) and type(val1) in [list, tuple]:
        _check_keys(val1, val2)
        applied_op = [{key: torch.mul(scalar, part) for key, part in val2.items()} for scalar in val1]
        output_class = type(val1)
    # Multiply val2 with something else
    elif not isinstance(val1, GTensor) and isinstance(val2, GTensor):
        applied_op = {key: torch.mul(val1, val) for key, val in val2.items()}
        output_class = type(val2)
    else:
        raise ValueError('Neither class inherits from GTensor!')

    return output_class(applied_op)


def mul(val1, val2):
    return _dispatch_mul(val1, val2)


def add(val1, val2):
    return _dispatch_op(torch.add, val1, val2)


def sub(val1, val2):
    return _dispatch_op(torch.sub, val1, val2)


def div(val1, val2):
    raise NotImplementedError('Complex Division has not been implemented yet')
    # return __dispatch_divtype(torch.div, val1, val2)


def cat(reps_list):
    """
    Concatenate (direct sum) a :obj:`list` of :obj:`GTensor` representations (along the channel dimension).

    Parameters
    ----------
    reps_list : :obj:`list` of :obj:`GTensor`

    Return
    ------
    rep_cat : :obj:`GTensor`
        Direct sum of all :obj:`GTensor` in `reps_list`
    """
    all_keys = set().union(*[rep.keys() for rep in reps_list])
    reps_cat = {key: [rep[key] for rep in reps_list if key in list(rep.keys())] for key in all_keys}
    return reps_list[0].__class__({key: torch.cat(reps, dim=reps_list[0].cdim) for key, reps in reps_cat.items() if len(reps) > 0})


def mix(weights, rep):
    """
    Linearly mix representation.

    Parameters
    ----------
    rep : :obj:`GVec` or compatible
    weights : :obj:`GWeights` or compatible

    Return
    ------
    :obj:`GVec`
        Mixed direct sum of all :obj:`GVec` in `reps_list`
    """
    if rep.keys() != weights.keys():
        raise ValueError('Must have one mixing weight for each part of GVec!')

    if isinstance(rep, GVec):
        rep_mix = GVec({key: mix_zweight_zvec(weights[key], rep[key]) for key in weights.keys()})
    elif isinstance(rep, GScalar):
        rep_mix = GScalar({key: mix_zweight_zscalar(weights[key], rep[key]) for key in weights.keys()})
    elif isinstance(rep, GWeight):
        rep_mix = GWeight({key: mix_zweight_zvec(weights[key], rep[key]) for key in weights.keys()})
    elif isinstance(rep, GTensor):
        raise NotImplementedError('Mixing for object {} not yet implemented!'.format(type(rep)))
    else:
        raise ValueError('Mixing only implemented for GTensor subclasses!')

    return rep_mix


def cat_mix(weights, reps_list):
    """
    First concatenate (direct sum) and then linearly mix a :obj:`list` of
    :obj:`GVec` objects with :obj:`GWeights` weights.

    Parameters
    ----------
    reps_list : :obj:`list` of :obj:`GVec` or compatible
    weights : :obj:`GWeights` or compatible

    Return
    ------
    :obj:`GVec`
        Mixed direct sum of all :obj:`GVec` in `reps_list`
    """

    return mix(weights, cat(reps_list))


def apply_wigner(wigner_d, rep, side='left'):
    """
    Apply a Wigner-D rotation to a :obj:`GVec` representation
    """
    return GVec({key: rot.rotate_part(wigner_d[key], part, side=side) for key, part in rep.items()})
