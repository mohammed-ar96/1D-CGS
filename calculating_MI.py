import networkx as nx
import operator
import math
import numpy as np
from scipy import stats
from collections import Counter
from scipy.sparse import identity, csr_matrix
from scipy.sparse.linalg import inv


def monotonicity_squared(R):
    rank_counts = Counter(R)
    num_nodes = sum(rank_counts.values())

    numerator = sum(count * (count - 1) for count in rank_counts.values())
    denominator = num_nodes * (num_nodes - 1)

    if denominator != 0:
        M_R = (1 - numerator / denominator) ** 2
    else:
        M_R = 0  # Handle case when |V| <= 1 to avoid division by zero

    return M_R

bc = np.loadtxt("MO_BC.dat")
dc = np.loadtxt("MO_DC.dat")
hi = np.loadtxt("MO_HI.dat")
kcore = np.loadtxt("MO_Kcore.dat")
vc = np.loadtxt("MO_VC.dat")
mdd = np.loadtxt("MO_MDD.dat")
nd = np.loadtxt("MO_ND.dat")
mrcnn = np.loadtxt("MO_MRCNN.dat")
rcnn = np.loadtxt("MO_RCNN.dat")
cgs = np.loadtxt("MO_CGS.dat")


print("Monotonicity (MI) using BC:", monotonicity_squared(list(bc)))
print("Monotonicity (MI) using DC:", monotonicity_squared(list(dc)))
print("Monotonicity (MI) using HI:", monotonicity_squared(list(hi)))
print("Monotonicity (MI) using K-CORE:", monotonicity_squared(list(kcore)))
print("Monotonicity (MI) using Vc:", monotonicity_squared(list(vc)))
print("Monotonicity (MI) using MDD:", monotonicity_squared(list(mdd)))
print("Monotonicity (MI) using ND:", monotonicity_squared(list(nd)))
print("Monotonicity (MI) using MRCNN:", monotonicity_squared(list(mrcnn)))
print("Monotonicity (MI) using RCNN:", monotonicity_squared(list(rcnn)))
print("Monotonicity (MI) using 1D-CGS:", monotonicity_squared(list(cgs)))