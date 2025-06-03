import statistics
import numpy as np

bc = np.loadtxt("kendall_BC.dat")
dc = np.loadtxt("kendall_DC.dat")
hi = np.loadtxt("kendall_HI.dat")
kcore = np.loadtxt("kendall_Kcore.dat")
vc = np.loadtxt("kendall_VC.dat")
mdd = np.loadtxt("kendall_MDD.dat")
nd = np.loadtxt("kendall_ND.dat")

mrcnn = np.loadtxt("kendall_MRCNN.dat")
rcnn = np.loadtxt("kendall_RCNN.dat")
cgs = np.loadtxt("kendall_CGS.dat")




print("Average of Kendall using BC:", statistics.mean(list(bc[:,1])))
print("Average of Kendall using DC:", statistics.mean(list(dc[:,1])))
print("Average of Kendall using HI:", statistics.mean(list(hi[:,1])))
print("Average of Kendall using K-CORE:", statistics.mean(list(kcore[:,1])))
print("Average of Kendall using Vc:", statistics.mean(list(vc[:,1])))
print("Average of Kendall using MDD:", statistics.mean(list(mdd[:,1])))
print("Average of Kendall using ND:", statistics.mean(list(nd[:,1])))
print("Average of Kendall using MRCNN:", statistics.mean(list(mrcnn[:,1])))
print("Average of Kendall using RCNN:", statistics.mean(list(rcnn[:,1])))
print("Average of Kendall using 1D-CGS:", statistics.mean(list(cgs[:,1])))
