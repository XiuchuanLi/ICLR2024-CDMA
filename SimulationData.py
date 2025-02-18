import numpy as np
import pandas as pd

def ToBij():
    if np.random.randn(1) < 0:
        signed = -1
    else:
        signed = 1
    return signed * np.random.uniform(.5, 2.0)


def Toa():
    return np.random.uniform(.5, 1)


def GaussData(Num=1000,seed=0):
    # Figure 1(a)
    np.random.seed(seed)
    noise = np.load(f'Gauss_{Num}.npy')[np.random.choice(np.arange(20), 17, replace=False)]

    L1=noise[0]*Toa()
    L2=noise[1]*Toa()+ToBij()*L1
    L3=noise[2]*Toa()+ToBij()*L1+ToBij()*L2
    O1=noise[3]*Toa()+ToBij()*L1
    O2=noise[4]*Toa()+ToBij()*L1
    O3=noise[5]*Toa()+ToBij()*L1
    O4=noise[6]*Toa()+ToBij()*L2
    O5=noise[7]*Toa()+ToBij()*L2
    O6=noise[8]*Toa()+ToBij()*L2
    O7=noise[9]*Toa()+ToBij()*L2+ToBij()*O6
    O8=noise[10]*Toa()+ToBij()*L3
    O9=noise[11]*Toa()+ToBij()*L3+ToBij()*O8
    O10=noise[12]*Toa()+ToBij()*L3
    O11=noise[13]*Toa()+ToBij()*L3
    O12=noise[14]*Toa()+ToBij()*L3+ToBij()*O11
    O13=noise[15]*Toa()+ToBij()*L1
    O14=noise[16]*Toa()+ToBij()*L2+ToBij()*L3+ToBij()*O13
    data = pd.DataFrame(np.array([O1,O2,O3,O4,O5,O6,O7,O8,O9,O10,O11,O12,O13,O14]),index=[f'O{i}' for i in range(1,15)])
    return data


def nonGaussData(Num=1000,seed=0):
    # Figure 1(b)
    np.random.seed(seed)
    noise = np.load(f'nonGauss_{Num}.npy')[np.random.choice(np.arange(18), 15, replace=False)]

    L1=noise[0]*Toa()
    L2=noise[1]*Toa()+ToBij()*L1
    L3=noise[2]*Toa()+ToBij()*L2
    O1=noise[3]*Toa()+ToBij()*L1
    O2=noise[4]*Toa()+ToBij()*L1
    O3=noise[5]*Toa()+ToBij()*L1
    O4=noise[6]*Toa()+ToBij()*L1+ToBij()*O3
    O5=noise[7]*Toa()+ToBij()*L2
    O6=noise[8]*Toa()+ToBij()*L2+ToBij()*O5
    O7=noise[9]*Toa()+ToBij()*L3
    O8=noise[10]*Toa()+ToBij()*L3
    O9=noise[11]*Toa()+ToBij()*L3
    O10=noise[12]*Toa()+ToBij()*L2+ToBij()*L3
    O11=noise[13]*Toa()+ToBij()*L2+ToBij()*O10
    O12=noise[14]*Toa()+ToBij()*L1+ToBij()*O11
    data = pd.DataFrame(np.array([O1,O2,O3,O4,O5,O6,O7,O8,O9,O10,O11,O12]),index=[f'O{i}' for i in range(1,13)])
    return data