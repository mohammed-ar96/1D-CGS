# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 02:32:58 2025

@author: لينوفو
"""

import numpy as np
import torch
import Embeddings
import Utils
import Models
import networkx as nx
from scipy import stats
from torch_geometric.data import Data, Batch
import alg2
import os
import statistics
import matplotlib.pyplot as plt

def jaccard_similarity(rank,sir_list,k):
    """Calculate Jaccard similarity between top k nodes of two rankings."""
    
    jaccard_list=[]
    # Jaccard similarity for each SIR simulation
    for sir in sir_list:
        sir_sort = [i for i, j in sorted(sir.items(), key=lambda x: x[1], reverse=True)]
        sir_rank = np.array((sir_sort), dtype=float)
        # print(sir,'=')
        # print(sir_rank,'\n')
        
        sir_topk = set(sir_rank[:k]) #(top-k nodes of SIR)
        # print(sir_topk)
        rank = list(rank)[:k]  
        
        rank = set(rank)  
        # print(rank)
        # print('dc=',rank)
        # print('sir_topk=',sir_topk)
        
        jaccard = len(rank.intersection(sir_topk)) / len(rank.union(sir_topk))
        
       
        jaccard_list.append(jaccard)
        
    return jaccard_list




# dataset_SIR = Utils.load_sir_list('./SIR results/jazz/jazz_')
# rs = './ranked_nodes_results/jazz/'
# folder_path = './JS/jazz/'

# result_dict = {}
# print(folder_path)
# outf = open(folder_path+"JS.dat", "w")

# for i in range(10,310,10):
        
#     bc_rank= np.loadtxt(rs+'jacc_BC.dat')
#     cgs_rank= np.loadtxt(rs+'jacc_CGS.dat')
#     dc_rank= np.loadtxt(rs+'jacc_DC.dat')
#     hi_rank= np.loadtxt(rs+'jacc_HI.dat')
#     kcore_rank= np.loadtxt(rs+'jacc_Kcore.dat')
#     mrcnn_rank= np.loadtxt(rs+'jacc_MRCNN.dat')
#     mdd_rank= np.loadtxt(rs+'jacc_MDD.dat')
#     rcnn_rank= np.loadtxt(rs+'jacc_RCNN.dat')
#     vc_rank= np.loadtxt(rs+'jacc_VC.dat')
#     nd_rank= np.loadtxt(rs+'jacc_ND.dat')
               
#     bc =jaccard_similarity(bc_rank,dataset_SIR,i)
#     cgs =jaccard_similarity(cgs_rank,dataset_SIR,i)
#     dc =jaccard_similarity(dc_rank,dataset_SIR,i)
#     hi =jaccard_similarity(hi_rank,dataset_SIR,i)
#     kcore =jaccard_similarity(kcore_rank,dataset_SIR,i)
#     mrcnn =jaccard_similarity(mrcnn_rank,dataset_SIR,i)
#     mdd =jaccard_similarity(mdd_rank,dataset_SIR,i)
#     rcnn =jaccard_similarity(rcnn_rank,dataset_SIR,i)
#     vc =jaccard_similarity(vc_rank,dataset_SIR,i)
#     nd =jaccard_similarity(nd_rank,dataset_SIR,i)
    
#     outf.write(str(i) + "  " + str(statistics.mean(bc))+ "  " + str(statistics.mean(dc)) + "  " + str(statistics.mean(hi))+ "  " 
#                           + str(statistics.mean(kcore))+ "  " + str(statistics.mean(vc))+ "  " + str(statistics.mean(mrcnn))+ "  " + str(statistics.mean(mdd))+ "  " + 
#                           str(statistics.mean(rcnn))+ "  " + str(statistics.mean(cgs))+ "  " + str(statistics.mean(nd)) + "\n")

# # print()
# outf.close()



s=['BC =','DC =','HI = ','K-CORE =','VC =','MRCNN =','MDD =','RCNN =','1D-CGS =','ND =']

y= np.loadtxt('./JS/facebook/JS.dat')
fb_bc=y[:,1]
fb_cgs=y[:,9]
fb_dc=y[:,2]
fb_hi=y[:,3]
fb_kcore=y[:,4]
fb_mrcnn=y[:,6]
fb_mdd=y[:,7]
fb_rcnn=y[:,8]
fb_vc=y[:,5]
fb_nd=y[:,10]



y= np.loadtxt('./JS/lastFM/JS.dat')
fm_bc=y[:,1]
fm_cgs=y[:,9]
fm_dc=y[:,2]
fm_hi=y[:,3]
fm_kcore=y[:,4]
fm_mrcnn=y[:,6]
fm_mdd=y[:,7]
fm_rcnn=y[:,8]
fm_vc=y[:,5]
fm_nd=y[:,10]




y= np.loadtxt('./JS/email/JS.dat')
em_bc=y[:,1]
em_cgs=y[:,9]
em_dc=y[:,2]
em_hi=y[:,3]
em_kcore=y[:,4]
em_mrcnn=y[:,6]
em_mdd=y[:,7]
em_rcnn=y[:,8]
em_vc=y[:,5]
em_nd=y[:,10]





y= np.loadtxt('./JS/epa/JS.dat')
ea_bc=y[:,1]
ea_cgs=y[:,9]
ea_dc=y[:,2]
ea_hi=y[:,3]
ea_kcore=y[:,4]
ea_mrcnn=y[:,6]
ea_mdd=y[:,7]
ea_rcnn=y[:,8]
ea_vc=y[:,5]
ea_nd=y[:,10]




y= np.loadtxt('./JS/figeys/JS.dat')
fg_bc=y[:,1]
fg_cgs=y[:,9]
fg_dc=y[:,2]
fg_hi=y[:,3]
fg_kcore=y[:,4]
fg_mrcnn=y[:,6]
fg_mdd=y[:,7]
fg_rcnn=y[:,8]
fg_vc=y[:,5]
fg_nd=y[:,10]



y= np.loadtxt('./JS/ham/JS.dat')
hm_bc=y[:,1]
hm_cgs=y[:,9]
hm_dc=y[:,2]
hm_hi=y[:,3]
hm_kcore=y[:,4]
hm_mrcnn=y[:,6]
hm_mdd=y[:,7]
hm_rcnn=y[:,8]
hm_vc=y[:,5]
hm_nd=y[:,10]







y= np.loadtxt('./JS/pgp/JS.dat')
pg_bc=y[:,1]
pg_cgs=y[:,9]
pg_dc=y[:,2]
pg_hi=y[:,3]
pg_kcore=y[:,4]
pg_mrcnn=y[:,6]
pg_mdd=y[:,7]
pg_rcnn=y[:,8]
pg_vc=y[:,5]
pg_nd=y[:,10]

 

y= np.loadtxt('./JS/router/JS.dat')
ro_bc=y[:,1]
ro_cgs=y[:,9]
ro_dc=y[:,2]
ro_hi=y[:,3]
ro_kcore=y[:,4]
ro_mrcnn=y[:,6]
ro_mdd=y[:,7]
ro_rcnn=y[:,8]
ro_vc=y[:,5]
ro_nd=y[:,10]



y= np.loadtxt('./JS/sex/JS.dat')
sx_bc=y[:,1]
sx_cgs=y[:,9]
sx_dc=y[:,2]
sx_hi=y[:,3]
sx_kcore=y[:,4]
sx_mrcnn=y[:,6]
sx_mdd=y[:,7]
sx_rcnn=y[:,8]
sx_vc=y[:,5]
sx_nd=y[:,10]




y= np.loadtxt('./JS/faa/JS.dat')
fa_bc=y[:,1]
fa_cgs=y[:,9]
fa_dc=y[:,2]
fa_hi=y[:,3]
fa_kcore=y[:,4]
fa_mrcnn=y[:,6]
fa_mdd=y[:,7]
fa_rcnn=y[:,8]
fa_vc=y[:,5]
fa_nd=y[:,10]




y= np.loadtxt('./JS/grq/JS.dat')
grq_bc=y[:,1]
grq_cgs=y[:,9]
grq_dc=y[:,2]
grq_hi=y[:,3]
grq_kcore=y[:,4]
grq_mrcnn=y[:,6]
grq_mdd=y[:,7]
grq_rcnn=y[:,8]
grq_vc=y[:,5]
grq_nd=y[:,10]






y= np.loadtxt('./JS/stelzl/JS.dat')
stel_bc=y[:,1]
stel_cgs=y[:,9]
stel_dc=y[:,2]
stel_hi=y[:,3]
stel_kcore=y[:,4]
stel_mrcnn=y[:,6]
stel_mdd=y[:,7]
stel_rcnn=y[:,8]
stel_vc=y[:,5]
stel_nd=y[:,10]



y= np.loadtxt('./JS/jazz/JS.dat')
jazz_bc=y[:,1]
jazz_cgs=y[:,9]
jazz_dc=y[:,2]
jazz_hi=y[:,3]
jazz_kcore=y[:,4]
jazz_mrcnn=y[:,6]
jazz_mdd=y[:,7]
jazz_rcnn=y[:,8]
jazz_vc=y[:,5]
jazz_nd=y[:,10]

# for i in range(1,11,1):
#     print(s[i-1],statistics.mean(y[:,i]))





a_list =np.arange(10,310,10)   
s=5

plt.figure(figsize=(20,19),dpi=120)

# plt.style.use('seaborn-ticks')
# plt.rcParams['xtick.direction'] = 'in'
# plt.rcParams['ytick.direction'] = 'in'
# font_size = 26
# plt.rc('xtick', labelsize=font_size)
# plt.rc('ytick', labelsize=font_size)

# plt.rcParams['font.family'] = 'serif'  
# plt.rcParams['font.serif'] = ['Arial']  

plt.subplot(4,3,1)
plt.plot(a_list,jazz_cgs,marker='o',markersize=s,c='red',label='1D-CGS')
plt.plot(a_list,jazz_dc,marker='<',markersize=s,c='fuchsia',label='DC')
plt.plot(a_list,jazz_rcnn,marker='o',markersize=s,c='b',label='RCNN')
plt.plot(a_list,jazz_kcore,marker='>',markersize=s,c='crimson',label='K-core')
plt.plot(a_list,jazz_mrcnn,marker='^',markersize=s,c='g',label='MRCNN')
plt.plot(a_list,jazz_nd,marker='p',markersize=s,c='y',label='ND')
plt.plot(a_list,jazz_bc,marker='h',markersize=s,c='black',label='BC')
plt.plot(a_list,jazz_vc,marker='H',markersize=s,c='orange',label='Vc')
plt.plot(a_list,jazz_mdd,marker='d',markersize=s,c='green',label='MDD')
plt.plot(a_list,jazz_hi,marker='*',markersize=s,c='purple',label='HI')
plt.title('Jazz',fontsize=20,fontweight='bold')
plt.ylabel('JS',fontsize=20,fontweight='bold')
plt.yticks(np.arange(0,1.1,0.2),fontsize=18)
plt.xticks(np.arange(0,350,50),fontsize=18)

plt.subplot(4,3,2)
plt.plot(a_list,em_cgs,marker='o',markersize=s,c='red',label='1D-CGS')
plt.plot(a_list,em_dc,marker='<',markersize=s,c='fuchsia',label='DC')
plt.plot(a_list,em_rcnn,marker='o',markersize=s,c='b',label='RCNN')
plt.plot(a_list,em_kcore,marker='>',markersize=s,c='crimson',label='K-core')
plt.plot(a_list,em_mrcnn,marker='^',markersize=s,c='g',label='MRCNN')
plt.plot(a_list,em_nd,marker='p',markersize=s,c='y',label='ND')
plt.plot(a_list,em_bc,marker='h',markersize=s,c='black',label='BC')
plt.plot(a_list,em_vc,marker='H',markersize=s,c='orange',label='Vc')
plt.plot(a_list,em_mdd,marker='d',markersize=s,c='green',label='MDD')
plt.plot(a_list,em_hi,marker='*',markersize=s,c='purple',label='HI')
plt.title('Email',fontsize=20,fontweight='bold')
# plt.ylabel(r'$\tau$',fontsize=24,fontweight='bold')
plt.yticks(np.arange(0,1.1,0.2),fontsize=18)
plt.xticks(np.arange(0,350,50),fontsize=18)

plt.subplot(4,3,3)
plt.plot(a_list,stel_cgs,marker='o',markersize=s,c='red',label='1D-CGS')
plt.plot(a_list,stel_dc,marker='<',markersize=s,c='fuchsia',label='DC')
plt.plot(a_list,stel_rcnn,marker='o',markersize=s,c='b',label='RCNN')
plt.plot(a_list,stel_kcore,marker='>',markersize=s,c='crimson',label='K-core')
plt.plot(a_list,stel_mrcnn,marker='^',markersize=s,c='g',label='MRCNN')
plt.plot(a_list,stel_nd,marker='p',markersize=s,c='y',label='ND')
plt.plot(a_list,stel_bc,marker='h',markersize=s,c='black',label='BC')
plt.plot(a_list,stel_vc,marker='H',markersize=s,c='orange',label='Vc')
plt.plot(a_list,stel_mdd,marker='d',markersize=s,c='green',label='MDD')
plt.plot(a_list,stel_hi,marker='*',markersize=s,c='purple',label='HI')
plt.title('Stelzl',fontsize=20,fontweight='bold')
# plt.ylabel('Jaccard Similarity',fontsize=44,fontweight='bold')
plt.yticks(np.arange(0,1.1,0.2),fontsize=18)
plt.xticks(np.arange(0,350,50),fontsize=18)
# plt.text(0.97,0.94,'(d)',fontsize=20,fontweight='bold')

plt.subplot(4,3,4)
plt.plot(a_list,fg_cgs,marker='o',markersize=s,c='red',label='1D-CGS')
plt.plot(a_list,fg_dc,marker='<',markersize=s,c='fuchsia',label='DC')
plt.plot(a_list,fg_rcnn,marker='o',markersize=s,c='b',label='RCNN')
plt.plot(a_list,fg_kcore,marker='>',markersize=s,c='crimson',label='K-core')
plt.plot(a_list,fg_mrcnn,marker='^',markersize=s,c='g',label='MRCNN')
plt.plot(a_list,fg_nd,marker='p',markersize=s,c='y',label='ND')
plt.plot(a_list,fg_bc,marker='h',markersize=s,c='black',label='BC')
plt.plot(a_list,fg_vc,marker='H',markersize=s,c='orange',label='Vc')
plt.plot(a_list,fg_mdd,marker='d',markersize=s,c='green',label='MDD')
plt.plot(a_list,fg_hi,marker='*',markersize=s,c='purple',label='HI')
plt.title('Figeys',fontsize=20,fontweight='bold')
plt.ylabel('JS',fontsize=20,fontweight='bold')
plt.yticks(np.arange(0,1.1,0.2),fontsize=18)
plt.xticks(np.arange(0,350,50),fontsize=18)
# plt.text(0.97,0.94,'(d)',fontsize=20,fontweight='bold')

plt.subplot(4,3,5)
plt.plot(a_list,hm_cgs,marker='o',markersize=s,c='red',label='1D-CGS')
plt.plot(a_list,hm_dc,marker='<',markersize=s,c='fuchsia',label='DC')
plt.plot(a_list,hm_rcnn,marker='o',markersize=s,c='b',label='RCNN')
plt.plot(a_list,hm_kcore,marker='>',markersize=s,c='crimson',label='K-core')
plt.plot(a_list,hm_mrcnn,marker='^',markersize=s,c='g',label='MRCNN')
plt.plot(a_list,hm_nd,marker='p',markersize=s,c='y',label='ND')
plt.plot(a_list,hm_bc,marker='h',markersize=s,c='black',label='BC')
plt.plot(a_list,hm_vc,marker='H',markersize=s,c='orange',label='Vc')
plt.plot(a_list,hm_mdd,marker='d',markersize=s,c='green',label='MDD')
plt.plot(a_list,hm_hi,marker='*',markersize=s,c='purple',label='HI')
plt.title('Hamster',fontsize=20,fontweight='bold')
plt.yticks(np.arange(0,1.1,0.2),fontsize=18)
plt.xticks(np.arange(0,350,50),fontsize=18)

plt.subplot(4,3,6)
plt.plot(a_list,fb_cgs,marker='o',markersize=s,c='red',label='1D-CGS')
plt.plot(a_list,fb_dc,marker='<',markersize=s,c='fuchsia',label='DC')
plt.plot(a_list,fb_rcnn,marker='o',markersize=s,c='b',label='RCNN')
plt.plot(a_list,fb_kcore,marker='>',markersize=s,c='crimson',label='K-core')
plt.plot(a_list,fb_mrcnn,marker='^',markersize=s,c='g',label='MRCNN')
plt.plot(a_list,fb_nd,marker='p',markersize=s,c='y',label='ND')
plt.plot(a_list,fb_bc,marker='h',markersize=s,c='black',label='BC')
plt.plot(a_list,fb_vc,marker='H',markersize=s,c='orange',label='Vc')
plt.plot(a_list,fb_mdd,marker='d',markersize=s,c='green',label='MDD')
plt.plot(a_list,fb_hi,marker='*',markersize=s,c='purple',label='HI')
plt.yticks(np.arange(0,1.1,0.2),fontsize=18)
plt.xticks(np.arange(0,350,50),fontsize=18)
plt.title('Facebook',fontsize=20,fontweight='bold')
# plt.text(0.97,0.94,'(f)',fontsize=20,fontweight='bold')

plt.subplot(4,3,7)
plt.plot(a_list,ea_cgs,marker='o',markersize=s,c='red',label='1D-CGS')
plt.plot(a_list,ea_dc,marker='<',markersize=s,c='fuchsia',label='DC')
plt.plot(a_list,ea_rcnn,marker='o',markersize=s,c='b',label='RCNN')
plt.plot(a_list,ea_kcore,marker='>',markersize=s,c='crimson',label='K-core')
plt.plot(a_list,ea_mrcnn,marker='^',markersize=s,c='g',label='MRCNN')
plt.plot(a_list,ea_nd,marker='p',markersize=s,c='y',label='ND')
plt.plot(a_list,ea_bc,marker='h',markersize=s,c='black',label='BC')
plt.plot(a_list,ea_vc,marker='H',markersize=s,c='orange',label='Vc')
plt.plot(a_list,ea_mdd,marker='d',markersize=s,c='green',label='MDD')
plt.plot(a_list,ea_hi,marker='*',markersize=s,c='purple',label='HI')
plt.title('EPA',fontsize=20,fontweight='bold')
plt.ylabel('JS',fontsize=20,fontweight='bold')
plt.yticks(np.arange(0,1.1,0.2),fontsize=18)
plt.xticks(np.arange(0,350,50),fontsize=18)
# plt.text(0.97,0.94,'(g)',fontsize=20,fontweight='bold')

plt.subplot(4,3,8)
plt.plot(a_list,ro_cgs,marker='o',markersize=s,c='red',label='1D-CGS')
plt.plot(a_list,ro_dc,marker='<',markersize=s,c='fuchsia',label='DC')
plt.plot(a_list,ro_rcnn,marker='o',markersize=s,c='b',label='RCNN')
plt.plot(a_list,ro_kcore,marker='>',markersize=s,c='crimson',label='K-core')
plt.plot(a_list,ro_mrcnn,marker='^',markersize=s,c='g',label='MRCNN')
plt.plot(a_list,ro_nd,marker='p',markersize=s,c='y',label='ND')
plt.plot(a_list,ro_bc,marker='h',markersize=s,c='black',label='BC')
plt.plot(a_list,ro_vc,marker='H',markersize=s,c='orange',label='Vc')
plt.plot(a_list,ro_mdd,marker='d',markersize=s,c='green',label='MDD')
plt.plot(a_list,ro_hi,marker='*',markersize=s,c='purple',label='HI')
plt.title('Router',fontsize=20,fontweight='bold')
# plt.ylabel(r'$\tau$',fontsize=38,fontweight='bold')
plt.yticks(np.arange(0,1.1,0.2),fontsize=18)
plt.xticks(np.arange(0,350,50),fontsize=18)
# plt.text(0.97,0.94,'(h)',fontsize=20,fontweight='bold')

plt.subplot(4,3,9)
plt.plot(a_list,grq_cgs,marker='o',markersize=s,c='red',label='1D-CGS')
plt.plot(a_list,grq_dc,marker='<',markersize=s,c='fuchsia',label='DC')
plt.plot(a_list,grq_rcnn,marker='o',markersize=s,c='b',label='RCNN')
plt.plot(a_list,grq_kcore,marker='>',markersize=s,c='crimson',label='K-core')
plt.plot(a_list,grq_mrcnn,marker='^',markersize=s,c='g',label='MRCNN')
plt.plot(a_list,grq_nd,marker='p',markersize=s,c='y',label='ND')
plt.plot(a_list,grq_bc,marker='h',markersize=s,c='black',label='BC')
plt.plot(a_list,grq_vc,marker='H',markersize=s,c='orange',label='Vc')
plt.plot(a_list,grq_mdd,marker='d',markersize=s,c='green',label='MDD')
plt.plot(a_list,grq_hi,marker='*',markersize=s,c='purple',label='HI')
plt.title('GrQ',fontsize=20,fontweight='bold')
# plt.ylabel(r'$\tau$',fontsize=38,fontweight='bold')
plt.yticks(np.arange(0,1.1,0.2),fontsize=18)
plt.xticks(np.arange(0,350,50),fontsize=18)
# plt.text(0.97,0.94,'(h)',fontsize=20,fontweight='bold')

plt.subplot(4,3,10)
plt.plot(a_list,fm_cgs,marker='o',markersize=s,c='red',label='1D-CGS')
plt.plot(a_list,fm_dc,marker='<',markersize=s,c='fuchsia',label='DC')
plt.plot(a_list,fm_rcnn,marker='o',markersize=s,c='b',label='RCNN')
plt.plot(a_list,fm_kcore,marker='>',markersize=s,c='crimson',label='K-core')
plt.plot(a_list,fm_mrcnn,marker='^',markersize=s,c='g',label='MRCNN')
plt.plot(a_list,fm_nd,marker='p',markersize=s,c='y',label='ND')
plt.plot(a_list,fm_bc,marker='h',markersize=s,c='black',label='BC')
plt.plot(a_list,fm_vc,marker='H',markersize=s,c='orange',label='Vc')
plt.plot(a_list,fm_mdd,marker='d',markersize=s,c='green',label='MDD')
plt.plot(a_list,fm_hi,marker='*',markersize=s,c='purple',label='HI')
plt.title('LastFM',fontsize=20,fontweight='bold')
plt.yticks(np.arange(0,1.1,0.2),fontsize=18)
plt.xticks(np.arange(0,350,50),fontsize=18)
plt.xlabel('Top k Nodes',fontsize=20,fontweight='bold')
plt.ylabel('JS',fontsize=20,fontweight='bold')
# plt.text(0.97,0.94,'(j)',fontsize=20,fontweight='bold')



plt.subplot(4,3,11)
plt.plot(a_list,pg_cgs,marker='o',markersize=s,c='red',label='1D-CGS')
plt.plot(a_list,pg_dc,marker='<',markersize=s,c='fuchsia',label='DC')
plt.plot(a_list,pg_rcnn,marker='o',markersize=s,c='b',label='RCNN')
plt.plot(a_list,pg_kcore,marker='>',markersize=s,c='crimson',label='K-core')
plt.plot(a_list,pg_mrcnn,marker='^',markersize=s,c='g',label='MRCNN')
plt.plot(a_list,pg_nd,marker='p',markersize=s,c='y',label='ND')
plt.plot(a_list,pg_bc,marker='h',markersize=s,c='black',label='BC')
plt.plot(a_list,pg_vc,marker='H',markersize=s,c='orange',label='Vc')
plt.plot(a_list,pg_mdd,marker='d',markersize=s,c='green',label='MDD')
plt.plot(a_list,pg_hi,marker='*',markersize=s,c='purple',label='HI')
plt.title('PGP',fontsize=20,fontweight='bold')
# plt.ylabel(r'$\tau$',fontsize=40,fontweight='bold')
plt.yticks(np.arange(0,1.1,0.2),fontsize=18)
plt.xticks(np.arange(0,350,50),fontsize=18)
plt.xlabel('Top k Nodes',fontsize=20,fontweight='bold')
# plt.text(0.97,0.94,'(k)',fontsize=20,fontweight='bold')

plt.subplot(4,3,12)
plt.plot(a_list,sx_cgs,marker='o',markersize=s,c='red',label='1D-CGS')
plt.plot(a_list,sx_dc,marker='<',markersize=s,c='fuchsia',label='DC')
plt.plot(a_list,sx_rcnn,marker='o',markersize=s,c='b',label='RCNN')
plt.plot(a_list,sx_kcore,marker='>',markersize=s,c='crimson',label='K-core')
plt.plot(a_list,sx_mrcnn,marker='^',markersize=s,c='g',label='MRCNN')
plt.plot(a_list,sx_nd,marker='p',markersize=s,c='y',label='ND')
plt.plot(a_list,sx_bc,marker='h',markersize=s,c='black',label='BC')
plt.plot(a_list,sx_vc,marker='H',markersize=s,c='orange',label='Vc')
plt.plot(a_list,sx_mdd,marker='d',markersize=s,c='green',label='MDD')
plt.plot(a_list,sx_hi,marker='*',markersize=s,c='purple',label='HI')
plt.title('Sex',fontsize=20,fontweight='bold')
# plt.ylabel(r'$\tau$',fontsize=40,fontweight='bold')
plt.yticks(np.arange(0,1.1,0.2),fontsize=18)
plt.xticks(np.arange(0,350,50),fontsize=18)
plt.xlabel('Top k Nodes',fontsize=20,fontweight='bold')
# plt.text(0.97,0.94,'(l)',fontsize=38,fontweight='bold')

# plt.subplots_adjust(hspace=0.7, wspace=5)
plt.tight_layout()
plt.legend(bbox_to_anchor=(0.25,-0.2),ncol=5,fontsize=18)
plt.savefig('jacc_plots.pdf',dpi=300, format='pdf', bbox_inches='tight')
plt.show()

