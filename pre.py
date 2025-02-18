from utils import Mergelist
import numpy as np
import scipy.stats as stats
from itertools import combinations

def TetradBasic(x1, x2, x3, x4, significance=0.01):
    # whether ({x1, x2}, {x3, x4}) satisfies the tetrad constraint
    # DAVID A. KENNY, A Test for a Vanishing Tetrad: The Second Canonical Correlations Equals Zero. Social Science Research, 1974.
    nSample = len(x1)
    r12 = np.corrcoef(x1, x2)[0, 1]
    r13 = np.corrcoef(x1, x3)[0, 1]
    r14 = np.corrcoef(x1, x4)[0, 1]
    r23 = np.corrcoef(x2, x3)[0, 1]
    r24 = np.corrcoef(x2, x4)[0, 1]
    r34 = np.corrcoef(x3, x4)[0, 1]

    a = (1-r12**2)*(1-r34**2)
    b = (r14-r12*r24)*(r13*r34-r14)+(r23-r13*r12)*(r24*r34-r23)+(r13-r12*r23)*(r14*r34-r13)+(r24-r14*r12)*(r23*r34-r24)
    c = (r13*r24-r14*r23)**2
    delta = b * b - 4 * a * c
    chi2 = -(nSample - 3.5) * np.log(1-(-b - delta**0.5)/(2*a))
    p_value = stats.chi2.sf(chi2, df=1)
    flag = np.int8(p_value > significance)
    return flag


def TetradMatrix(X):
    dim = X.shape[0]
    matrix = np.ones([dim, dim, dim, dim])
    for (i, j) in combinations(range(dim), 2):
        for(k, l) in combinations([x for x in range(dim) if x not in [i, j]], 2):
            vanish = TetradBasic(X[i], X[j], X[k], X[l])
            matrix[i, j, k, l] = vanish
            matrix[i, j, l, k] = vanish
            matrix[j, i, k, l] = vanish
            matrix[j, i, l, k] = vanish
    return matrix


def GenePurePairs(X):
    dim = X.shape[0]
    matrix = TetradMatrix(X)
    local_index = [x for x in range(dim) if x not in [0, 5]]
    clusters = []
    for (i, j) in combinations(range(dim), 2):
        local_index = [x for x in range(dim) if x not in [i, j]]
        local_tetrad_matrix = matrix[i, j][local_index,:][:,local_index]
        if np.all(local_tetrad_matrix == 1):
            clusters.append([i, j])
    return matrix, Mergelist(clusters)
