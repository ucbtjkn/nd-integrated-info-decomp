import scipy
import numpy as np
from itertools import product
from collections import Counter

def local_entropy_mvn(x, mu, cov):
    return -np.log(scipy.stats.multivariate_normal.pdf(x, mu, cov, allow_singular=True))


def local_entropy_discrete(x):
    
    if x.ndim == 1:
        x = x[None, :]

    n_dim, n_samp = x.shape
    combs = list(product([n for n in range(np.max(x + 1))], repeat=n_dim))
    distri = list(zip(*x.tolist()))
    c = Counter(distri)
    p = np.array([c.get(comb, 0) for comb in combs]) / n_samp
    entropy_dict = {comb: -np.log2(p_) for comb, p_ in zip(combs, p)}

    return np.array([entropy_dict[comb] for comb in distri])


def double_redundancy_mmi(mutual_information, mi_means, atom):
    atom_mis = mi_means[atom[0],:] [:,atom[1]]
    (i1,i2) = np.unravel_index(np.argmin(atom_mis), shape=(len(atom[0]),len(atom[1])))
    return mutual_information[atom[0][i1],atom[1][i2],:]

def double_redundancy_ccs():
    return