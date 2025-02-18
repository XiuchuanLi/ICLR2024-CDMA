import numpy as np
from itertools import combinations
from utils import independence


def SelectPdf(Num,data_type="exp-non-gaussian"):
    if data_type == "exp-non-gaussian":
        noise = np.random.uniform(-1, 1, size=Num) ** 7

    elif data_type == "laplace":
        noise =np.random.laplace(0, 1, size=Num)

    elif data_type == "exponential":
        noise = np.random.exponential(scale=1.0, size=Num)

    else: #gaussian
        noise = np.random.normal(0, 1, size=Num)

    return noise


def normalize(data):
    data -= np.mean(data)
    data /= np.std(data)
    return data


noises = []
for i in range(20):
    print(i)
    while True:
        new_noise = normalize(SelectPdf(1000, "gaussian"))
        if np.all(np.array([np.abs(np.corrcoef(new_noise, noise)[0, 1]) < 0.02 for noise in noises])):
            noises.append(new_noise)
            break
noises = np.stack(noises, axis=0)
np.save(f'Gauss_1000.npy', noises)

noises = []
for i in range(18):
    print(i)
    while True:
        new_noise = normalize(SelectPdf(1000, "exp-non-gaussian"))
        if np.all(np.array([np.abs(np.corrcoef(new_noise, noise)[0, 1]) < 0.025 for noise in noises])) \
           and np.all(np.array([independence(new_noise, noise, 0.2)[0] for noise in noises])):
            noises.append(new_noise)
            break
noises = np.stack(noises, axis=0)
np.save(f'nonGauss_1000.npy', noises)
