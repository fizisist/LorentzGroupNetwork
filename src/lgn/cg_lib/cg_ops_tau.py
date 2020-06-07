from math import inf

from lgn.g_lib import GTau


def cg_product_tau(tau1, tau2, maxdim=inf):
    """
    Calulate output multiplicity of the CG Product of two G Vectors
    given the multiplicty of two input G Vectors.

    Parameters
    ----------
    tau1 : :class:`list` of :class:`int`, :class:`GTau`.
        Multiplicity of first representation.

    tau2 : :class:`list` of :class:`int`, :class:`GTau`.
        Multiplicity of second representation.

    maxdim : :class:`int`
        Largest weight to include in CG Product.

    Return
    ------

    tau : :class:`GTau`
        Multiplicity of output representation.

    """
    tau1 = GTau(tau1)
    tau2 = GTau(tau2)
    tau = {}

    for (k1, n1) in tau1.keys():
        for (k2, n2) in tau2.keys():
            if max(k1, n1, k2, n2) >= maxdim:
                continue
            kmin, kmax = abs(k1 - k2), min(k1 + k2, maxdim - 1)
            nmin, nmax = abs(n1 - n2), min(n1 + n2, maxdim - 1)
            for k in range(kmin, kmax + 1, 2):
                for n in range(nmin, nmax + 1, 2):
                    tau.setdefault((k, n), 0)
                    tau[(k, n)] += tau1[(k1, n1)] * tau2[(k2, n2)]

    return GTau(tau)
