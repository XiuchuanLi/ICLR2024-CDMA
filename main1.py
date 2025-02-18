import numpy as np
from SimulationData import GaussData
from pre import GenePurePairs

def Pairs(X):
    matrix, clusters = GenePurePairs(X)
    pairs = []
    for i, cluster in enumerate(clusters):
        if len(cluster) > 2:
            pairs.append((cluster, 'pure', None))
        else:
            pure = True
            i, j = cluster[0], cluster[1]
            for k in [x for x in range(len(matrix)) if x not in [i, j]]:
                local_index = [x for x in range(len(matrix)) if x not in [i, j, k]]
                local_tetrad_matrix = matrix[i, k][local_index,:][:,local_index]
                if np.all(local_tetrad_matrix == 1):
                    pairs.append((cluster, 'impure', k)) # k is Ref(cluster)
                    pure = False
                    break
            if pure:
                pairs.append((cluster, 'pure', None))
    return pairs


def Latents(pairs):
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
            if impure_pair[-1] in results[key][0][0] or impure_pair[-1] == results[key][0][-1]:
                results[key].append(impure_pair)
                new_latent = False
                break
        if new_latent:
            num_latent += 1
            results[f'l{num_latent}'] = [impure_pair,]
    return results


for seed in range(10):
    print(seed)
    data = GaussData(1000, seed).values
    pairs = Pairs(data)
    latents = Latents(pairs)
    print(latents)