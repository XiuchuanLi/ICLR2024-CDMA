import numpy as np
from SimulationData import nonGaussData
from itertools import permutations
from pre import TetradBasic, GenePurePairs
from utils import independence

def constraint(x1, x2, x3, x4):
    A = np.array([[np.cov(x1, x2)[0, 1], np.cov(x1, x3)[0, 1]], [np.cov(x2, x4)[0, 1], np.cov(x3, x4)[0, 1]]])
    b = -np.array([[np.cov(x1,x1)[0, 1]], [np.cov(x1,x4)[0, 1]]])
    coefficient = (np.linalg.inv(A) @ b).flatten().tolist()
    return independence(x1 + coefficient[0] * x2 + coefficient[1] * x3, x1)[0], coefficient[0]

def Pairs(X):
    # All observed variables are candidate variables
    matrix, clusters = GenePurePairs(X)
    pairs = []
    for i, cluster in enumerate(clusters):
        if len(cluster) > 2:
            pairs.append((cluster, 'pure', np.inf))
        else:
            pure = True
            i, j = cluster[0], cluster[1]
            for(k, l) in permutations([x for x in range(len(matrix)) if x not in [i, j]], 2):
                if constraint(X[i], X[j], X[k], X[l])[0]:
                    alpha = constraint(X[i], X[j], X[k], X[l])[1]
                    if np.abs(alpha) > 2: # the direct causal strength less than 1/2
                        continue
                    pairs.append(([i, j], 'impure', alpha))
                    pure = False
                    break
                elif constraint(X[j], X[i], X[k], X[l])[0]:
                    alpha = constraint(X[j], X[i], X[k], X[l])[1]
                    if np.abs(alpha) > 2: # the direct causal strength less than 1/2
                        continue
                    pairs.append(([j, i], 'impure', alpha))
                    pure = False
                    break
            if pure:
                pairs.append((cluster, 'pure', np.inf))
    return pairs


def Latents(X, pairs):
    pure_pairs = [elem for elem in pairs if elem[1] == 'pure']
    impure_pairs = [elem for elem in pairs if elem[1] == 'impure']
    num_latent = 0
    results = {}
    for pure_pair in pure_pairs:
        num_latent += 1
        results[f'l{num_latent}'] = [pure_pair,]
    for impure_pair in impure_pairs:
        new_latent = True
        for key in results:
            if results[key][0][1] == 'pure':
                x1, x3 = X[results[key][0][0][0]], X[results[key][0][0][1]]
            else:
                x1, x3 = X[results[key][0][0][0]], X[results[key][0][0][1]] + 1 / results[key][0][-1] * X[results[key][0][0][0]]
            x2, x4 = X[impure_pair[0][0]], X[impure_pair[0][1]] + 1 / impure_pair[-1] * X[impure_pair[0][0]]
            if TetradBasic(x1,x2,x3,x4,0.05):
                results[key].append(impure_pair)
                new_latent = False
                break
        if new_latent:
            num_latent += 1
            results[f'l{num_latent}'] = [impure_pair,]
    return results


for seed in range(10, 20):
    print(seed)
    data = nonGaussData(1000, seed).values
    pairs = Pairs(data)
    latents = Latents(data, pairs)
    print(latents)
