import torch
import numpy as np
from lgn.cg_lib import CGDict

def rotate_part(D, z, side='left', autoconvert=True, conjugate=False):
    """ Apply a D matrix using complex broadcast matrix multiplication. """
    if autoconvert:
        D = D.to(z.device, z.dtype)
    if conjugate:
        D = dagger(D)
    Dr, Di = D.unbind(0)
    zr, zi = z.unbind(0)

    if side == 'left':
        return torch.stack((torch.matmul(zr, Dr) + torch.matmul(zi, Di),
                            - torch.matmul(zr, Di) + torch.matmul(zi, Dr)), 0)
    elif side == 'right':
        return torch.stack((torch.matmul(Dr, zr) + torch.matmul(Di, zi),
                            - torch.matmul(Di, zr) + torch.matmul(Dr, zi)), 0)
    else:
        raise ValueError('Must choose side: left/right.')


def rotate_rep(rep, alpha, beta, gamma, side='left', conjugate=False, cg_dict=None):
    """ Apply a part-wise left/right sided D-matrix to a (matrix) representation. """
    device, dtype = rep.device, rep.dtype
    return rep.__class__({key: rotate_part(LorentzD(key, alpha, beta, gamma, cg_dict=cg_dict, device=device, dtype=dtype), part, side=side, conjugate=conjugate) for key, part in rep.items()})


def create_J(j):
    mrange = -np.arange(-j, j)
    jp_diag = np.sqrt((j + mrange) * (j - mrange + 1))
    Jp = np.diag(jp_diag, k=1)
    Jm = np.diag(jp_diag, k=-1)
    # Jx = (Jp + Jm) / complex(2, 0)
    # Jy = -(Jp - Jm) / complex(0, 2)
    Jz = np.diag(-np.arange(-j, j + 1))
    Id = np.eye(2 * j + 1)
    return Jp, Jm, Jz, Id


def create_Jy(j):
    mrange = -np.arange(-j, j)
    jp_diag = np.sqrt((j + mrange) * (j - mrange + 1))
    Jp = np.diag(jp_diag, k=1)
    Jm = np.diag(jp_diag, k=-1)
    Jy = -(Jp - Jm) / complex(0, 2)
    return Jy


def create_Jx(j):
    mrange = -np.arange(-j, j)
    jp_diag = np.sqrt((j + mrange) * (j - mrange + 1))
    Jp = np.diag(jp_diag, k=1)
    Jm = np.diag(jp_diag, k=-1)
    Jx = (Jp + Jm) / complex(2, 0)
    return Jx


def littled(j, beta):
    Jy = create_Jy(j)
    evals, evecs = np.linalg.eigh(Jy)
    evecsh = evecs.conj().T
    evals_exp = np.diag(np.exp(1j * beta * evals))
    d = np.matmul(np.matmul(evecs, evals_exp), evecsh)
    return d


def WignerD(j, alpha, beta, gamma, numpy_test=False, dtype=torch.float, device=torch.device('cpu')):
    d = littled(j, beta)

    Jz = np.arange(-j, j + 1)
    Jzl = np.expand_dims(Jz, 1)

    # np.multiply() broadcasts, so this isn't actually matrix multiplication, and 'left'/'right' are lies
    left = np.exp(1j * alpha * Jzl)
    right = np.exp(1j * gamma * Jz)

    D = left * d * right

    if not numpy_test:
        D = complex_from_numpy(D, dtype=dtype, device=device)

    return D


def LorentzD(key, alpha, beta, gamma, numpy_test=False, dtype=torch.float, device=torch.device('cpu'), cg_dict=None):

    (k, n) = key
    if cg_dict is None:
        cg_dict = CGDict(maxdim=max(k, n) + 1, transpose=True, dtype=dtype, device=device)._cg_dict

    D = complex_tensor_prod(WignerD(k / 2, alpha, beta, gamma, numpy_test=numpy_test, dtype=dtype, device=device),
                            conj(WignerD(n / 2, -alpha, beta, -gamma, numpy_test=numpy_test, dtype=dtype, device=device)))
    cg_mat = cg_dict[((k, 0), (0, n))][(k, n)]
    D_re = torch.matmul(torch.matmul(cg_mat, D.unbind(0)[0]), cg_mat.t())
    D_im = torch.matmul(torch.matmul(cg_mat, D.unbind(0)[1]), cg_mat.t())
    D = torch.stack((D_re, D_im), 0)
    return D


def dagger(D):
    conj = torch.tensor([1, -1], dtype=D.dtype, device=D.device).view(2, 1, 1)
    D = (D * conj).permute((0, 2, 1))
    return D


def conj(D):
    conj = torch.tensor([1, -1], dtype=D.dtype, device=D.device).view(2, 1, 1)
    D = D * conj
    return D


def complex_from_numpy(z, dtype=torch.float, device=torch.device('cpu')):
    """ Take a numpy array and output a complex array of the same size. """
    zr = torch.from_numpy(z.real).to(dtype=dtype, device=device)
    zi = torch.from_numpy(z.imag).to(dtype=dtype, device=device)

    return torch.stack((zr, zi), 0)


def complex_tensor_prod(d1, d2):
    d1_re, d1_im = d1.unbind(0)
    d2_re, d2_im = d2.unbind(0)
    s1 = d1.shape[1:]
    s2 = d2.shape[1:]
    assert len(s1) == 2 and len(
        s2) == 2, "Both tensors must be of rank 2 (and complex)!"
    d_re = d1_re.view(s1[0], 1, s1[1], 1) * d2_re.view(1, s2[0], 1, s2[1]) - \
        d1_im.view(s1[0], 1, s1[1], 1) * d2_im.view(1, s2[0], 1, s2[1])
    d_im = d1_re.view(s1[0], 1, s1[1], 1) * d2_im.view(1, s2[0], 1, s2[1]) + \
        d1_im.view(s1[0], 1, s1[1], 1) * d2_re.view(1, s2[0], 1, s2[1])
    return torch.stack((d_re, d_im), 0).view(2, s1[0] * s2[0], s1[1] * s2[1])
