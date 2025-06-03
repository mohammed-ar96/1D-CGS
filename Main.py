# -*- coding: utf-8 -*-
"""
Created on Sat March  1 21:34:06 2025

@author: لينوفو
"""
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from tqdm import tqdm
from scipy import stats
import Utils
import alg2
import Test
import Models
import Embeddings
import os
import RT
import time




    

warnings.filterwarnings('ignore')
sns.set_style('ticks')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
setup_seed(5)

if __name__ == '__main__':
    
    L1 = 28                        
    batch_size = 32
    num_epochs =500
    lr = 0.001

    
        # Load training network
    BA_1000_4 = Utils.load_graph('./Networks/training/Train_1000_4.txt')
    BA_1000_4_sir = pd.read_csv('./SIR results/Train_1000_4/BA_1000_4.csv')
    BA_1000_4_label = dict(zip(np.array(BA_1000_4_sir['Node'],dtype=str),BA_1000_4_sir['SIR']))
    
    # ===============================================================
    # load real-world networks
    LastFM = Utils.load_graph('./Networks/real/LastFM.txt')
    Ham = Utils.load_graph('./Networks/real/Peh_edge.txt')
    Facebook = Utils.load_graph('./Networks/real/facebook_combined.txt')
    Figeys = Utils.load_graph('./Networks/real/figeys.txt')
    Sex = Utils.load_graph('./Networks/real/sex.txt')
    Email= Utils.load_graph('./Networks/real/email.txt')
    Router = Utils.load_graph('./Networks/real/router.txt')
    pgp = Utils.load_graph('./Networks/real/pgp.txt')
    EPA = Utils.load_graph('./Networks/real/epa.txt')
    GrQ = Utils.load_graph('./Networks/real/CA-GrQc.txt')
    Jazz = Utils.load_graph('./Networks/real/jazz.txt')
    Stelzl = Utils.load_graph('./Networks/real/stelzl.txt')
    # ==============================================================
    # remove selfloops of real-world networks
    LastFM.remove_edges_from(nx.selfloop_edges(LastFM))
    Figeys.remove_edges_from(nx.selfloop_edges(Figeys))
    Email.remove_edges_from(nx.selfloop_edges(Email))
    Router.remove_edges_from(nx.selfloop_edges(Router))
    pgp.remove_edges_from(nx.selfloop_edges(pgp))
    EPA.remove_edges_from(nx.selfloop_edges(EPA))
    GrQ.remove_edges_from(nx.selfloop_edges(GrQ))
    Vidal.remove_edges_from(nx.selfloop_edges(Vidal))
    Jazz.remove_edges_from(nx.selfloop_edges(Jazz))
    Stelzl.remove_edges_from(nx.selfloop_edges(Stelzl))
    # ========================================================================
    # load labels of real-world networks
    LastFM_SIR = Utils.load_sir_list('./SIR results/LastFM/LastFM_')
    Ham_SIR = Utils.load_sir_list('./SIR results/Ham/Ham_')
    Facebook_SIR = Utils.load_sir_list('./SIR results/Facebook/Facebook_')
    GrQ_SIR = Utils.load_sir_list('./SIR results/GrQ/GrQ_')
    Jazz_SIR = Utils.load_sir_list('./SIR results/jazz/jazz_')
    Stelzl_SIR = Utils.load_sir_list('./SIR results/stelzl/stelzl_')
    Figeys_SIR = Utils.load_sir_list('./SIR results/Figeys/Figeys_')
    Sex_SIR = Utils.load_sir_list('./SIR results/Sex/Sex_')
    Email_SIR = Utils.load_sir_list('./SIR results/Email/Email_')
    Router_SIR = Utils.load_sir_list('./SIR results/router/sir_')
    pgp_SIR = Utils.load_sir_list('./SIR results/pgp/pgp_')
    EPA_SIR = Utils.load_sir_list('./SIR results/epa/epa_')
    # ==================================================
    # community division
    _,BA_1000_4_community,_ = Utils.Louvain(BA_1000_4)
    _,Facebook_community,_ = Utils.Louvain(Facebook)
    _,Ham_community,_ = Utils.Louvain(Ham)
    _,LastFM_community,_ = Utils.Louvain(LastFM)
    _,Sex_community,_ = Utils.Louvain(Sex)
    _,Figeys_community,_ = Utils.Louvain(Figeys)
    _,GrQ_community,_ = Utils.Louvain(GrQ)
    _,Jazz_community,_ = Utils.Louvain(Jazz)
    _,Stelzl_community,_ = Utils.Louvain(Stelzl)
    _,Email_community,_ = Utils.Louvain(Email)
    _,Router_community,_ = Utils.Louvain(Router)
    _,pgp_community,_ = Utils.Louvain(pgp)
    _,EPA_community,_ = Utils.Louvain(EPA)
    
    
    # # =================================================================================================
    # # Prepare data and train 1D-CGS model
    # features, edge_index, node_to_idx = Embeddings.main2(BA_1000_4)
    # nodes = list(BA_1000_4.nodes())
    # labels = torch.tensor([BA_1000_4_label[node] for node in nodes], dtype=torch.float)
    # model = Utils.train_1DCGS_model(features, edge_index, labels)# Train model
    # print("1D-CGS model training complete!\n")
    # # ===========================================================================
    
   
    # # Initialize and train MRCNN Model
    # # BA_1000_mrcnn = Embeddings.main1(BA_1000_4,L1,BA_1000_4_community)
    # # mrcnn_loader = Utils.Get_DataLoader1(BA_1000_mrcnn,BA_1000_4_label,batch_size,L1)
    # # mrcnn= Models.CNN1(L1)
    # # MRCNN,MRCNN_loss = Utils.train_model(mrcnn_loader,mrcnn,num_epochs,lr,L1)
    
    # # torch.save(MRCNN.state_dict(), 'mrcnn_model.pth')
    
    # MRCNN = Models.CNN1(L1)
    # MRCNN.load_state_dict(torch.load('./mrcnn_model.pth'))
    # # ==========================================================================
    
    # # Initialize and train RCNN Model
    # # BA_1000_rcnn = Embeddings.main(BA_1000_4,L1)
    # # rcnn_loader = Utils.Get_DataLoader(BA_1000_rcnn,BA_1000_4_label,batch_size,L1)
    # # rcnn= Models.CNN0(L1)
    # # RCNN,RCNN_loss = Utils.train_model(rcnn_loader,rcnn,num_epochs,lr,L1,path='rcnn_model.pth')
    # # print("RCNN model training complete!")
    
    # # torch.save(RCNN.state_dict(), 'rcnn_model.pth')
    
    # RCNN = Models.CNN0(L1)
    # RCNN.load_state_dict(torch.load('./rcnn_model.pth'))
    
    # ==========================================================================
    
    
# #     # Evaluate on real network datasets

#     LastFM_CGS_tau,LastFM_RCNN_tau, LastFM_dc_tau, LastFM_ks_tau, LastFM_nd_tau, LastFM_bc_tau, LastFM_vc_tau, LastFM_MDD_tau, LastFM_HI_tau, LastFM_KSIF_tau = Test.compare_tau(LastFM, model, LastFM_SIR, LastFM_community,RCNN,MRCNN,'./kendall_result/lastFM/','./ranked_nodes_results/lastFM/')

#     Ham_CGS_tau,Ham_RCNN_tau, Ham_dc_tau, Ham_ks_tau, Ham_nd_tau, Ham_bc_tau, Ham_vc_tau , Ham_MDD_tau, Ham_HI_tau, Ham_KSIF_tau= Test.compare_tau(Ham, model, Ham_SIR, Ham_community,RCNN,MRCNN,'./kendall_result/hamster/','./ranked_nodes_results/hamster/')
  
#     Facebook_CGS_tau,Facebook_RCNN_tau, Facebook_dc_tau, Facebook_ks_tau, Facebook_nd_tau, Facebook_bc_tau, Facebook_vc_tau , Facebook_MDD_tau, Facebook_HI_tau, Facebook_KSIF_tau= Test.compare_tau(Facebook, model, Facebook_SIR, Facebook_community,RCNN,MRCNN,'./kendall_result/Facebook/','./ranked_nodes_results/Facebook/')
  
#     Sex_CGS_tau,Sex_RCNN_tau, Sex_dc_tau, Sex_ks_tau, Sex_nd_tau, Sex_bc_tau, Sex_vc_tau , Sex_MDD_tau, Sex_HI_tau, Sex_KSIF_tau= Test.compare_tau(Sex, model, Sex_SIR, Sex_community,RCNN,MRCNN,'./kendall_result/Sex/','./ranked_nodes_results/sex/')
  
#     Figeys_CGS_tau,Figeys_RCNN_tau, Figeys_dc_tau, Figeys_ks_tau, Figeys_nd_tau, Figeys_bc_tau, Figeys_vc_tau , Figeys_MDD_tau, Figeys_HI_tau, Figeys_KSIF_tau= Test.compare_tau(Figeys, model, Figeys_SIR, Figeys_community,RCNN,MRCNN,'./kendall_result/Figeys/','./ranked_nodes_results/Figeys/')
  
#     GrQ_CGS_tau,GrQ_RCNN_tau, GrQ_dc_tau, GrQ_ks_tau, GrQ_nd_tau, GrQ_bc_tau, GrQ_vc_tau , GrQ_MDD_tau, GrQ_HI_tau, GrQ_KSIF_tau= Test.compare_tau(GrQ, model, GrQ_SIR, GrQ_community,RCNN,MRCNN,'./kendall_result/grq/','./ranked_nodes_results/grq/')

#     Email_CGS_tau,Email_RCNN_tau, Email_dc_tau, Email_ks_tau, Email_nd_tau, Email_bc_tau, Email_vc_tau , Email_MDD_tau, Email_HI_tau, Email_KSIF_tau= Test.compare_tau(Email, model, Email_SIR, Email_community,RCNN,MRCNN,'./kendall_result/email/','./ranked_nodes_results/email/')

#     Jazz_CGS_tau,Jazz_RCNN_tau, Jazz_dc_tau, Jazz_ks_tau, Jazz_nd_tau, Jazz_bc_tau, Jazz_vc_tau , Jazz_MDD_tau, Jazz_HI_tau, Jazz_KSIF_tau= Test.compare_tau(Jazz, model, Jazz_SIR, Jazz_community,RCNN,MRCNN,'./kendall_result/Jazz/','./ranked_nodes_results/jazz/')

#     Stelzl_CGS_tau,Stelzl_RCNN_tau, Stelzl_dc_tau, Stelzl_ks_tau, Stelzl_nd_tau, Stelzl_bc_tau, Stelzl_vc_tau , Stelzl_MDD_tau, Stelzl_HI_tau, Stelzl_KSIF_tau= Test.compare_tau(Stelzl, model, Stelzl_SIR, Stelzl_community,RCNN,MRCNN,'./kendall_result/Stelzl/','./ranked_nodes_results/stelzl/')
# # 
#     Router_CGS_tau,Router_RCNN_tau, Router_dc_tau, Router_ks_tau, Router_nd_tau, Router_bc_tau, Router_vc_tau , Router_MDD_tau, Router_HI_tau, Router_KSIF_tau= Test.compare_tau(Router, model, Router_SIR, Router_community,RCNN,MRCNN,'./kendall_result/Router/','./ranked_nodes_results/router/')
    
#     pgp_CGS_tau,pgp_RCNN_tau, pgp_dc_tau, pgp_ks_tau, pgp_nd_tau, pgp_bc_tau, pgp_vc_tau , pgp_MDD_tau, pgp_HI_tau, pgp_KSIF_tau= Test.compare_tau(pgp, model, pgp_SIR, pgp_community,RCNN,MRCNN,'./kendall_result/pgp/','./ranked_nodes_results/pgp/')

#     EPA_CGS_tau,EPA_RCNN_tau, EPA_dc_tau, EPA_ks_tau, EPA_nd_tau, EPA_bc_tau, EPA_vc_tau , EPA_MDD_tau, EPA_HI_tau, EPA_KSIF_tau= Test.compare_tau(EPA, model, EPA_SIR, EPA_community,RCNN,MRCNN,'./kendall_result/EPA/','./ranked_nodes_results/epa/')
    
# #     # ========================================================================================================================================
    
# #     # Plot results
#     a_list = np.arange(1,2,0.1)
#     plt.figure(figsize=(20,19),dpi=120)
#     size=10
#     # plt.style.use('seaborn-ticks')
#     # plt.rcParams['xtick.direction'] = 'in'
#     # plt.rcParams['ytick.direction'] = 'in'
#     # font_size = 26
#     # plt.rc('xtick', labelsize=font_size)
#     # plt.rc('ytick', labelsize=font_size)
    
#     # plt.rcParams['font.family'] = 'serif'  
#     # plt.rcParams['font.serif'] = ['Arial']  
    
#     plt.subplot(4,3,1)
#     plt.plot(a_list,Jazz_CGS_tau,marker='o',markersize=size,c='red',label='1D-CGS')
#     plt.plot(a_list,Jazz_dc_tau,marker='<',markersize=size,c='fuchsia',label='DC')
#     plt.plot(a_list,Jazz_RCNN_tau,marker='o',markersize=size,c='b',label='RCNN')
#     plt.plot(a_list,Jazz_ks_tau,marker='>',markersize=size,c='crimson',label='K-core')
#     plt.plot(a_list,Jazz_KSIF_tau,marker='^',markersize=size,c='g',label='MRCNN')
#     plt.plot(a_list,Jazz_nd_tau,marker='p',markersize=size,c='y',label='ND')
#     plt.plot(a_list,Jazz_bc_tau,marker='h',markersize=size,c='black',label='BC')
#     plt.plot(a_list,Jazz_vc_tau,marker='H',markersize=size,c='orange',label='Vc')
#     plt.plot(a_list,Jazz_MDD_tau,marker='d',markersize=size,c='green',label='MDD')
#     plt.plot(a_list,Jazz_HI_tau,marker='*',markersize=size,c='purple',label='HI')
#     plt.title('Jazz',fontsize=20,fontweight='bold')
#     plt.ylabel(r'$\tau$',fontsize=30,fontweight='bold')
#     plt.yticks(np.arange(0,1.1,0.2),fontsize=18)
#     plt.xticks(np.arange(1,2,0.3),fontsize=18)
#     # plt.xlabel(r'$\mu/\mu_{\mathrm{th}}$',fontsize=38,fontweight='bold')
#     # plt.text(0.97,0.94,'(a)',fontsize=20,fontweight='bold')
    
#     plt.subplot(4,3,2)
#     plt.plot(a_list,Email_CGS_tau,marker='o',markersize=size,c='red',label='1D-CGS')
#     plt.plot(a_list,Email_dc_tau,marker='<',markersize=size,c='fuchsia',label='DC')
#     plt.plot(a_list,Email_RCNN_tau,marker='o',markersize=size,c='b',label='RCNN')
#     plt.plot(a_list,Email_ks_tau,marker='>',markersize=size,c='crimson',label='K-core')
#     plt.plot(a_list,Email_KSIF_tau,marker='^',markersize=size,c='g',label='MRCNN')
#     plt.plot(a_list,Email_nd_tau,marker='p',markersize=size,c='y',label='ND')
#     plt.plot(a_list,Email_bc_tau,marker='h',markersize=size,c='black',label='BC')
#     plt.plot(a_list,Email_vc_tau,marker='H',markersize=size,c='orange',label='Vc')
#     plt.plot(a_list,Email_MDD_tau,marker='d',markersize=size,c='green',label='MDD')
#     plt.plot(a_list,Email_HI_tau,marker='*',markersize=size,c='purple',label='HI')
#     plt.title('Email',fontsize=20,fontweight='bold')
#     # plt.ylabel(r'$\tau$',fontsize=24,fontweight='bold')
#     plt.yticks(np.arange(0,1.1,0.2),fontsize=18)
#     plt.xticks(np.arange(1,2,0.3),fontsize=18)
#     # plt.xlabel(r'$\mu/\mu_{\mathrm{th}}$,fontsize=38,fontweight='bold')
#     # plt.text(0.97,0.94,'(b)',fontsize=20,fontweight='bold')
    

#     plt.subplot(4,3,3)
#     plt.plot(a_list,Stelzl_CGS_tau,marker='o',markersize=size,c='red',label='1D-CGS')
#     plt.plot(a_list,Stelzl_dc_tau,marker='<',markersize=size,c='fuchsia',label='DC')
#     plt.plot(a_list,Stelzl_RCNN_tau,marker='o',markersize=size,c='b',label='RCNN')
#     plt.plot(a_list,Stelzl_ks_tau,marker='>',markersize=size,c='crimson',label='K-core')
#     plt.plot(a_list,Stelzl_KSIF_tau,marker='^',markersize=size,c='g',label='MRCNN')
#     plt.plot(a_list,Stelzl_nd_tau,marker='p',markersize=size,c='y',label='ND')
#     plt.plot(a_list,Stelzl_bc_tau,marker='h',markersize=size,c='black',label='BC')
#     plt.plot(a_list,Stelzl_vc_tau,marker='H',markersize=size,c='orange',label='Vc')
#     plt.plot(a_list,Stelzl_MDD_tau,marker='d',markersize=size,c='green',label='MDD')
#     plt.plot(a_list,Stelzl_HI_tau,marker='*',markersize=size,c='purple',label='HI')
#     plt.title('Stelzl',fontsize=20,fontweight='bold')
#     # plt.ylabel(r'$\tau$',fontsize=24,fontweight='bold')
#     plt.yticks(np.arange(0,1.1,0.2),fontsize=18)
#     plt.xticks(np.arange(1,2,0.3),fontsize=18)
#     # plt.text(0.97,0.94,'(c)',fontsize=20,fontweight='bold')
    
#     plt.subplot(4,3,4)
#     plt.plot(a_list,Figeys_CGS_tau,marker='o',markersize=size,c='red',label='1D-CGS')
#     plt.plot(a_list,Figeys_dc_tau,marker='<',markersize=size,c='fuchsia',label='DC')
#     plt.plot(a_list,Figeys_RCNN_tau,marker='o',markersize=size,c='b',label='RCNN')
#     plt.plot(a_list,Figeys_ks_tau,marker='>',markersize=size,c='crimson',label='K-core')
#     plt.plot(a_list,Figeys_KSIF_tau,marker='^',markersize=size,c='g',label='MRCNN')
#     plt.plot(a_list,Figeys_nd_tau,marker='p',markersize=size,c='y',label='ND')
#     plt.plot(a_list,Figeys_bc_tau,marker='h',markersize=size,c='black',label='BC')
#     plt.plot(a_list,Figeys_vc_tau,marker='H',markersize=size,c='orange',label='Vc')
#     plt.plot(a_list,Figeys_MDD_tau,marker='d',markersize=size,c='green',label='MDD')
#     plt.plot(a_list,Figeys_HI_tau,marker='*',markersize=size,c='purple',label='HI')
#     plt.title('Figeys',fontsize=20,fontweight='bold')
#     plt.ylabel(r'$\tau$',fontsize=30,fontweight='bold')
#     plt.yticks(np.arange(0,1.1,0.2),fontsize=18)
#     plt.xticks(np.arange(1,2,0.3),fontsize=18)
#     # plt.text(0.97,0.94,'(d)',fontsize=20,fontweight='bold')
    
#     plt.subplot(4,3,5)
#     plt.plot(a_list,Ham_CGS_tau,marker='o',markersize=size,c='red',label='1D-CGS')
#     plt.plot(a_list,Ham_dc_tau,marker='<',markersize=size,c='fuchsia',label='DC')
#     plt.plot(a_list,Ham_RCNN_tau,marker='o',markersize=size,c='b',label='RCNN')
#     plt.plot(a_list,Ham_ks_tau,marker='>',markersize=size,c='crimson',label='K-core')
#     plt.plot(a_list,Ham_KSIF_tau,marker='^',markersize=size,c='g',label='MRCNN')
#     plt.plot(a_list,Ham_nd_tau,marker='p',markersize=size,c='y',label='ND')
#     plt.plot(a_list,Ham_bc_tau,marker='h',markersize=size,c='black',label='BC')
#     plt.plot(a_list,Ham_vc_tau,marker='H',markersize=size,c='orange',label='Vc')
#     plt.plot(a_list,Ham_MDD_tau,marker='d',markersize=size,c='green',label='MDD')
#     plt.plot(a_list,Ham_HI_tau,marker='*',markersize=size,c='purple',label='HI')
#     plt.title('Hamster',fontsize=20,fontweight='bold')
#     plt.yticks(np.arange(0,1.1,0.2),fontsize=18)
#     plt.xticks(np.arange(1,2,0.3),fontsize=18)
#     # plt.xlabel(r'$\mu/\mu_{\mathrm{th}}$',fontsize=20,fontweight='bold')
#     # plt.text(0.97,0.94,'(e)',fontsize=20,fontweight='bold')
    

#     plt.subplot(4,3,6)
#     plt.plot(a_list,Facebook_CGS_tau,marker='o',markersize=size,c='red',label='1D-CGS')
#     plt.plot(a_list,Facebook_dc_tau,marker='<',markersize=size,c='fuchsia',label='DC')
#     plt.plot(a_list,Facebook_RCNN_tau,marker='o',markersize=size,c='b',label='RCNN')
#     plt.plot(a_list,Facebook_ks_tau,marker='>',markersize=size,c='crimson',label='K-core')
#     plt.plot(a_list,Facebook_KSIF_tau,marker='^',markersize=size,c='g',label='MRCNN')
#     plt.plot(a_list,Facebook_nd_tau,marker='p',markersize=size,c='y',label='ND')
#     plt.plot(a_list,Facebook_bc_tau,marker='h',markersize=size,c='black',label='BC')
#     plt.plot(a_list,Facebook_vc_tau,marker='H',markersize=size,c='orange',label='Vc')
#     plt.plot(a_list,Facebook_MDD_tau,marker='d',markersize=size,c='green',label='MDD')
#     plt.plot(a_list,Facebook_HI_tau,marker='*',markersize=size,c='purple',label='HI')
#     plt.yticks(np.arange(0,1.1,0.2),fontsize=18)
#     plt.xticks(np.arange(1,2,0.3),fontsize=18)
#     plt.title('Facebook',fontsize=20,fontweight='bold')
#     # plt.text(0.97,0.94,'(f)',fontsize=20,fontweight='bold')
    
    
#     plt.subplot(4,3,7)
#     plt.plot(a_list,EPA_CGS_tau,marker='o',markersize=size,c='red',label='1D-CGS')
#     plt.plot(a_list,EPA_dc_tau,marker='<',markersize=size,c='fuchsia',label='DC')
#     plt.plot(a_list,EPA_RCNN_tau,marker='o',markersize=size,c='b',label='RCNN')
#     plt.plot(a_list,EPA_ks_tau,marker='>',markersize=size,c='crimson',label='K-core')
#     plt.plot(a_list,EPA_KSIF_tau,marker='^',markersize=size,c='g',label='MRCNN')
#     plt.plot(a_list,EPA_nd_tau,marker='p',markersize=size,c='y',label='ND')
#     plt.plot(a_list,EPA_bc_tau,marker='h',markersize=size,c='black',label='BC')
#     plt.plot(a_list,EPA_vc_tau,marker='H',markersize=size,c='orange',label='Vc')
#     plt.plot(a_list,EPA_MDD_tau,marker='d',markersize=size,c='green',label='MDD')
#     plt.plot(a_list,EPA_HI_tau,marker='*',markersize=size,c='purple',label='HI')
#     plt.title('EPA',fontsize=20,fontweight='bold')
#     plt.ylabel(r'$\tau$',fontsize=30,fontweight='bold')
#     plt.yticks(np.arange(0,1.1,0.2),fontsize=18)
#     plt.xticks(np.arange(1,2,0.3),fontsize=18)
#     # plt.text(0.97,0.94,'(g)',fontsize=20,fontweight='bold')
    
#     plt.subplot(4,3,8)
#     plt.plot(a_list,Router_CGS_tau,marker='o',markersize=size,c='red',label='1D-CGS')
#     plt.plot(a_list,Router_dc_tau,marker='<',markersize=size,c='fuchsia',label='DC')
#     plt.plot(a_list,Router_RCNN_tau,marker='o',markersize=size,c='b',label='RCNN')
#     plt.plot(a_list,Router_ks_tau,marker='>',markersize=size,c='crimson',label='K-core')
#     plt.plot(a_list,Router_KSIF_tau,marker='^',markersize=size,c='g',label='MRCNN')
#     plt.plot(a_list,Router_nd_tau,marker='p',markersize=size,c='y',label='ND')
#     plt.plot(a_list,Router_bc_tau,marker='h',markersize=size,c='black',label='BC')
#     plt.plot(a_list,Router_vc_tau,marker='H',markersize=size,c='orange',label='Vc')
#     plt.plot(a_list,Router_MDD_tau,marker='d',markersize=size,c='green',label='MDD')
#     plt.plot(a_list,Router_HI_tau,marker='*',markersize=size,c='purple',label='HI')
#     plt.title('Router',fontsize=20,fontweight='bold')
#     # plt.ylabel(r'$\tau$',fontsize=38,fontweight='bold')
#     plt.yticks(np.arange(0,1.1,0.2),fontsize=18)
#     plt.xticks(np.arange(1,2,0.3),fontsize=18)
#     # plt.text(0.97,0.94,'(h)',fontsize=20,fontweight='bold')
    
#     plt.subplot(4,3,9)
#     plt.plot(a_list,GrQ_CGS_tau,marker='o',markersize=size,c='red',label='1D-CGS')
#     plt.plot(a_list,GrQ_dc_tau,marker='<',markersize=size,c='fuchsia',label='DC')
#     plt.plot(a_list,GrQ_RCNN_tau,marker='o',markersize=size,c='b',label='RCNN')
#     plt.plot(a_list,GrQ_ks_tau,marker='>',markersize=size,c='crimson',label='K-core')
#     plt.plot(a_list,GrQ_KSIF_tau,marker='^',markersize=size,c='g',label='MRCNN')
#     plt.plot(a_list,GrQ_nd_tau,marker='p',markersize=size,c='y',label='ND')
#     plt.plot(a_list,GrQ_bc_tau,marker='h',markersize=size,c='black',label='BC')
#     plt.plot(a_list,GrQ_vc_tau,marker='H',markersize=size,c='orange',label='Vc')
#     plt.plot(a_list,GrQ_MDD_tau,marker='d',markersize=size,c='green',label='MDD')
#     plt.plot(a_list,GrQ_HI_tau,marker='*',markersize=size,c='purple',label='HI')
#     plt.title('GrQ',fontsize=20,fontweight='bold')
#     # plt.ylabel(r'$\tau$',fontsize=38,fontweight='bold')
#     plt.yticks(np.arange(0,1.1,0.2),fontsize=18)
#     plt.xticks(np.arange(1,2,0.3),fontsize=18)
#     # plt.text(0.97,0.94,'(h)',fontsize=20,fontweight='bold')
    
#     plt.subplot(4,3,10)
#     plt.plot(a_list,LastFM_CGS_tau,marker='o',markersize=size,c='red',label='1D-CGS')
#     plt.plot(a_list,LastFM_dc_tau,marker='<',markersize=size,c='fuchsia',label='DC')
#     plt.plot(a_list,LastFM_RCNN_tau,marker='o',markersize=size,c='b',label='RCNN')
#     plt.plot(a_list,LastFM_ks_tau,marker='>',markersize=size,c='crimson',label='K-core')
#     plt.plot(a_list,LastFM_KSIF_tau,marker='^',markersize=size,c='g',label='MRCNN')
#     plt.plot(a_list,LastFM_nd_tau,marker='p',markersize=size,c='y',label='ND')
#     plt.plot(a_list,LastFM_bc_tau,marker='h',markersize=size,c='black',label='BC')
#     plt.plot(a_list,LastFM_vc_tau,marker='H',markersize=size,c='orange',label='Vc')
#     plt.plot(a_list,LastFM_MDD_tau,marker='d',markersize=size,c='green',label='MDD')
#     plt.plot(a_list,LastFM_HI_tau,marker='*',markersize=size,c='purple',label='HI')
#     plt.title('LastFM',fontsize=20,fontweight='bold')
#     plt.ylabel(r'$\tau$',fontsize=30,fontweight='bold')
#     plt.yticks(np.arange(0,1.1,0.2),fontsize=18)
#     plt.xticks(np.arange(1,2,0.3),fontsize=18)
#     # plt.text(0.97,0.94,'(i)',fontsize=20,fontweight='bold')
#     plt.xlabel(r'$\mu/\mu_{\mathrm{th}}$',fontsize=24,fontweight='bold')

    
#     plt.subplot(4,3,11)
#     plt.plot(a_list,pgp_CGS_tau,marker='o',markersize=size,c='red',label='1D-CGS')
#     plt.plot(a_list,pgp_dc_tau,marker='<',markersize=size,c='fuchsia',label='DC')
#     plt.plot(a_list,pgp_RCNN_tau,marker='o',markersize=size,c='b',label='RCNN')
#     plt.plot(a_list,pgp_ks_tau,marker='>',markersize=size,c='crimson',label='K-core')
#     plt.plot(a_list,pgp_KSIF_tau,marker='^',markersize=size,c='g',label='MRCNN')
#     plt.plot(a_list,pgp_nd_tau,marker='p',markersize=size,c='y',label='ND')
#     plt.plot(a_list,pgp_bc_tau,marker='h',markersize=size,c='black',label='BC')
#     plt.plot(a_list,pgp_vc_tau,marker='H',markersize=size,c='orange',label='Vc')
#     plt.plot(a_list,pgp_MDD_tau,marker='d',markersize=size,c='green',label='MDD')
#     plt.plot(a_list,pgp_HI_tau,marker='*',markersize=size,c='purple',label='HI')
#     plt.title('PGP',fontsize=20,fontweight='bold')
#     # plt.ylabel(r'$\tau$',fontsize=40,fontweight='bold')
#     plt.yticks(np.arange(0,1.1,0.2),fontsize=18)
#     plt.xticks(np.arange(1,2,0.3),fontsize=18)
#     plt.xlabel(r'$\mu/\mu_{\mathrm{th}}$',fontsize=24,fontweight='bold')
#     # plt.text(0.97,0.94,'(k)',fontsize=20,fontweight='bold')
    
#     plt.subplot(4,3,12)
#     plt.plot(a_list,Sex_CGS_tau,marker='o',markersize=size,c='red',label='1D-CGS')
#     plt.plot(a_list,Sex_dc_tau,marker='<',markersize=size,c='fuchsia',label='DC')
#     plt.plot(a_list,Sex_RCNN_tau,marker='o',markersize=size,c='b',label='RCNN')
#     plt.plot(a_list,Sex_ks_tau,marker='>',markersize=size,c='crimson',label='K-core')
#     plt.plot(a_list,Sex_KSIF_tau,marker='^',markersize=size,c='g',label='MRCNN')
#     plt.plot(a_list,Sex_nd_tau,marker='p',markersize=size,c='y',label='ND')
#     plt.plot(a_list,Sex_bc_tau,marker='h',markersize=size,c='black',label='BC')
#     plt.plot(a_list,Sex_vc_tau,marker='H',markersize=size,c='orange',label='Vc')
#     plt.plot(a_list,Sex_MDD_tau,marker='d',markersize=size,c='green',label='MDD')
#     plt.plot(a_list,Sex_HI_tau,marker='*',markersize=size,c='purple',label='HI')
#     plt.title('Sex',fontsize=20,fontweight='bold')
#     # plt.ylabel(r'$\tau$',fontsize=40,fontweight='bold')
#     plt.yticks(np.arange(0,1.1,0.2),fontsize=18)
#     plt.xticks(np.arange(1,2,0.3),fontsize=18)
#     plt.xlabel(r'$\mu/\mu_{\mathrm{th}}$',fontsize=24,fontweight='bold')
#     # plt.text(0.97,0.94,'(l)',fontsize=20,fontweight='bold')

#     # plt.subplots_adjust(hspace=0.7, wspace=5)
#     plt.tight_layout()
#     plt.legend(bbox_to_anchor=(0.25,-0.2),ncol=5,fontsize=18)
#     plt.savefig('tau_plots.pdf',dpi=300, format='pdf', bbox_inches='tight')
#     plt.show()
    
    
    
    
    
    
    
# # =====================================================================
    
# #     # calc rank of nodes for Jacc Similarity \ stored in ranked nodes results
  
#     LastFM = Test.calc_ranks(LastFM, model, LastFM_SIR, LastFM_community,RCNN,MRCNN,'./kendall_result/lastFM/','./ranked_nodes_results/lastFM/')

#     Ham= Test.calc_ranks(Ham, model, Ham_SIR, Ham_community,RCNN,MRCNN,'./kendall_result/hamster/','./ranked_nodes_results/hamster/')
  
#     Facebook= Test.calc_ranks(Facebook, model, Facebook_SIR, Facebook_community,RCNN,MRCNN,'./kendall_result/Facebook/','./ranked_nodes_results/Facebook/')
  
#     Sex= Test.calc_ranks(Sex, model, Sex_SIR, Sex_community,RCNN,MRCNN,'./kendall_result/Sex/','./ranked_nodes_results/sex/')
  
#     Figeys= Test.calc_ranks(Figeys, model, Figeys_SIR, Figeys_community,RCNN,MRCNN,'./kendall_result/Figeys/','./ranked_nodes_results/Figeys/')
  
#     GrQ= Test.calc_ranks(GrQ, model, GrQ_SIR, GrQ_community,RCNN,MRCNN,'./kendall_result/grq/','./ranked_nodes_results/grq/')
 
#     Email= Test.calc_ranks(Email, model, Email_SIR, Email_community,RCNN,MRCNN,'./kendall_result/email/','./ranked_nodes_results/email/')

#     Jazz= Test.calc_ranks(Jazz, model, Jazz_SIR, Jazz_community,RCNN,MRCNN,'./kendall_result/Jazz/','./ranked_nodes_results/jazz/')

#     Stelzl= Test.calc_ranks(Stelzl, model, Stelzl_SIR, Stelzl_community,RCNN,MRCNN,'./kendall_result/Stelzl/','./ranked_nodes_results/stelzl/')
# # 
#     Router= Test.calc_ranks(Router, model, Router_SIR, Router_community,RCNN,MRCNN,'./kendall_result/Router/','./ranked_nodes_results/router/')

#     pgp = Test.calc_ranks(pgp, model, pgp_SIR, pgp_community,RCNN,MRCNN,'./kendall_result/pgp/','./ranked_nodes_results/pgp/')

#     EPA= Test.calc_ranks(EPA, model, EPA_SIR, EPA_community,RCNN,MRCNN,'./kendall_result/EPA/','./ranked_nodes_results/epa/')
    
    
    
    
    
    
    
    
# ====================================================================================================================================
    # Calc node scores for MI\ stored in MO folder
#     Jazzmo= Test.calc_score(Jazz, model, Jazz_SIR, Jazz_community,RCNN,MRCNN,'./kendall_result/Jazz/','./MO/jazz/')

#     LastFMmo = Test.calc_score(LastFM, model, LastFM_SIR, LastFM_community,RCNN,MRCNN,'./kendall_result/lastFM/','./MO/lastFM/')

#     Hammo= Test.calc_score(Ham, model, Ham_SIR, Ham_community,RCNN,MRCNN,'./kendall_result/hamster/','./MO/hamster/')
  
#     Facebookmo= Test.calc_score(Facebook, model, Facebook_SIR, Facebook_community,RCNN,MRCNN,'./kendall_result/Facebook/','./MO/Facebook/')
  
#     Sexmo= Test.calc_score(Sex, model, Sex_SIR, Sex_community,RCNN,MRCNN,'./kendall_result/Sex/','./MO/sex/')
  
#     Figeysmo= Test.calc_score(Figeys, model, Figeys_SIR, Figeys_community,RCNN,MRCNN,'./kendall_result/Figeys/','./MO/Figeys/')
  
#     GrQmo= Test.calc_score(GrQ, model, GrQ_SIR, GrQ_community,RCNN,MRCNN,'./kendall_result/grq/','./MO/grq/')
  
#     Emailmo= Test.calc_score(Email, model, Email_SIR, Email_community,RCNN,MRCNN,'./kendall_result/email/','./MO/email/')
  
#     Stelzlmo= Test.calc_score(Stelzl, model, Stelzl_SIR, Stelzl_community,RCNN,MRCNN,'./kendall_result/Stelzl/','./MO/stelzl/')
# # 
#     Routermo= Test.calc_score(Router, model, Router_SIR, Router_community,RCNN,MRCNN,'./kendall_result/Router/','./MO/router/')
    
#     pgpmo = Test.calc_score(pgp, model, pgp_SIR, pgp_community,RCNN,MRCNN,'./kendall_result/pgp/','./MO/pgp/')

#     EPAmo= Test.calc_score(EPA, model, EPA_SIR, EPA_community,RCNN,MRCNN,'./kendall_result/EPA/','./MO/epa/')







# ===============================================================================================================================
#     # calc Running Time  \ stored in algorithm times file
 
#     LastFM = Test.compare_RT(LastFM, model, LastFM_SIR, LastFM_community,RCNN,MRCNN,'./kendall_result/lastFM/')

#     Ham= Test.compare_RT(Ham, model, Ham_SIR, Ham_community,RCNN,MRCNN,'./kendall_result/hamster/')
  
#     Facebook= Test.compare_RT(Facebook, model, Facebook_SIR, Facebook_community,RCNN,MRCNN,'./kendall_result/Facebook/')
  
#     Sex= Test.compare_RT(Sex, model, Sex_SIR, Sex_community,RCNN,MRCNN,'./kendall_result/Sex/')
  
#     Figeys= Test.compare_RT(Figeys, model, Figeys_SIR, Figeys_community,RCNN,MRCNN,'./kendall_result/Figeys/')
  
#     GrQ= Test.compare_RT(GrQ, model, GrQ_SIR, GrQ_community,RCNN,MRCNN,'./kendall_result/grq/')

#     Email= Test.compare_RT(Email, model, Email_SIR, Email_community,RCNN,MRCNN,'./kendall_result/email/')

#     Jazz= Test.compare_RT(Jazz, model, Jazz_SIR, Jazz_community,RCNN,MRCNN,'./kendall_result/Jazz/')

#     Vidal= Test.compare_RT(Vidal, model, Vidal_SIR, Vidal_community,RCNN,MRCNN,'./kendall_result/Vidal/')
    
#     Stelzl= Test.compare_RT(Stelzl, model, Stelzl_SIR, Stelzl_community,RCNN,MRCNN,'./kendall_result/Stelzl/')
# # 
#     Router= Test.compare_RT(Router, model, Router_SIR, Router_community,RCNN,MRCNN,'./kendall_result/Router/')
 
#     pgp = Test.compare_RT(pgp, model, pgp_SIR, pgp_community,RCNN,MRCNN,'./kendall_result/pgp/')

#     EPA= Test.compare_RT(EPA, model, EPA_SIR, EPA_community,RCNN,MRCNN,'./kendall_result/EPA/')
# ===============================================
    # L=28
    # # EXP: compare the R-Time of MRCNN, RCNN, and 1D-CGS
    # # Load training networks
    # BA_1000_4 = Utils.load_graph('./Networks/training/Train_1000_4.txt')
    # BA_1000_4_sir = pd.read_csv('./SIR results/Train_1000_4/BA_1000_4.csv')
    # BA_1000_4_label = dict(zip(np.array(BA_1000_4_sir['Node'],dtype=str),BA_1000_4_sir['SIR']))
    
    # BA_1000_10 = Utils.load_graph('./Networks/training/BA_1000_10.txt')
    # BA_1000_10_sir = pd.read_csv('./SIR results/BA_1000_10/BA_1000_10_5.csv')
    # BA_1000_10_label = dict(zip(np.array(BA_1000_10_sir['Node'],dtype=str),BA_1000_10_sir['SIR']))
    
    # BA_1000_20 = Utils.load_graph('./Networks/training/BA_1000_20.txt')
    # BA_1000_20_sir = pd.read_csv('./SIR results/BA_1000_20/BA_1000_20_5.csv')
    # BA_1000_20_label = dict(zip(np.array(BA_1000_20_sir['Node'],dtype=str),BA_1000_20_sir['SIR']))
    
    # BA_3000_4 = Utils.load_graph('./Networks/training/BA_3000_4.txt')
    # BA_3000_4_sir = pd.read_csv('./SIR results/BA_3000_4/BA_3000_4_5.csv')
    # BA_3000_4_label = dict(zip(np.array(BA_3000_4_sir['Node'],dtype=str),BA_3000_4_sir['SIR']))
    
    # BA_3000_10 = Utils.load_graph('./Networks/training/BA_3000_10.txt')
    # BA_3000_10_sir = pd.read_csv('./SIR results/BA_3000_10/BA_3000_10_5.csv')
    # BA_3000_10_label = dict(zip(np.array(BA_3000_10_sir['Node'],dtype=str),BA_3000_10_sir['SIR']))
    
    # BA_3000_20 = Utils.load_graph('./Networks/training/BA_3000_20.txt')
    # BA_3000_20_sir = pd.read_csv('./SIR results/BA_3000_20/BA_3000_20_5.csv')
    # BA_3000_20_label = dict(zip(np.array(BA_3000_20_sir['Node'],dtype=str),BA_3000_20_sir['SIR']))
    
    # BA_4000_4 = Utils.load_graph('./Networks/training/BA_4000_4.txt')
    # BA_4000_4_sir = pd.read_csv('./SIR results/BA_4000_4/BA_4000_4_5.csv')
    # BA_4000_4_label = dict(zip(np.array(BA_4000_4_sir['Node'],dtype=str),BA_4000_4_sir['SIR']))
    
    # BA_4000_10 = Utils.load_graph('./Networks/training/BA_4000_10.txt')
    # BA_4000_10_sir = pd.read_csv('./SIR results/BA_4000_10/BA_4000_10_5.csv')
    # BA_4000_10_label = dict(zip(np.array(BA_4000_10_sir['Node'],dtype=str),BA_4000_10_sir['SIR']))
    
    # BA_4000_20 = Utils.load_graph('./Networks/training/BA_4000_20.txt')
    # BA_4000_20_sir = pd.read_csv('./SIR results/BA_4000_20/BA_4000_20_5.csv')
    # BA_4000_20_label = dict(zip(np.array(BA_4000_20_sir['Node'],dtype=str),BA_4000_20_sir['SIR']))
    
    # RCNN_train_time_1000_4 = Test.cal_training_time(BA_1000_4, BA_1000_4_label, L, batch_size, num_epochs, lr)
    # RCNN_train_time_1000_10 = Test.cal_training_time(BA_1000_10, BA_1000_10_label, L, batch_size, num_epochs, lr)
    # RCNN_train_time_1000_20 = Test.cal_training_time(BA_1000_20, BA_1000_20_label, L, batch_size, num_epochs, lr)
    # RCNN_train_time_3000_4 = Test.cal_training_time(BA_3000_4, BA_3000_4_label, L, batch_size, num_epochs, lr)
    # RCNN_train_time_3000_10 = Test.cal_training_time(BA_3000_10, BA_3000_10_label, L, batch_size, num_epochs, lr)
    # RCNN_train_time_3000_20 = Test.cal_training_time(BA_3000_20, BA_3000_20_label, L, batch_size, num_epochs, lr)
    # RCNN_train_time_4000_4 = Test.cal_training_time(BA_4000_4, BA_4000_4_label, L, batch_size, num_epochs, lr)
    # RCNN_train_time_4000_10 = Test.cal_training_time(BA_4000_10, BA_4000_10_label, L, batch_size, num_epochs, lr)
    # RCNN_train_time_4000_20 = Test.cal_training_time(BA_4000_20, BA_4000_20_label, L, batch_size, num_epochs, lr)
    

    # MRCNN_train_time_1000_4 = Test.cal_training_time2(BA_1000_4, BA_1000_4_label, L, batch_size, num_epochs, lr)
    # MRCNN_train_time_1000_10 = Test.cal_training_time2(BA_1000_10, BA_1000_10_label, L, batch_size, num_epochs, lr)
    # MRCNN_train_time_1000_20 = Test.cal_training_time2(BA_1000_20, BA_1000_20_label, L, batch_size, num_epochs, lr)
    # MRCNN_train_time_3000_4 = Test.cal_training_time2(BA_3000_4, BA_3000_4_label, L, batch_size, num_epochs, lr)
    # MRCNN_train_time_3000_10 = Test.cal_training_time2(BA_3000_10, BA_3000_10_label, L, batch_size, num_epochs, lr)
    # MRCNN_train_time_3000_20 = Test.cal_training_time2(BA_3000_20, BA_3000_20_label, L, batch_size, num_epochs, lr)
    # MRCNN_train_time_4000_4 = Test.cal_training_time2(BA_4000_4, BA_4000_4_label, L, batch_size, num_epochs, lr)
    # MRCNN_train_time_4000_10 = Test.cal_training_time2(BA_4000_10, BA_4000_10_label, L, batch_size, num_epochs, lr)
    # MRCNN_train_time_4000_20 = Test.cal_training_time2(BA_4000_20, BA_4000_20_label, L, batch_size, num_epochs, lr)
    
    # MRCNN_time_train_pd = pd.DataFrame({
    #         'MRCNN_1000_4': [MRCNN_train_time_1000_4],
    #         'MRCNN_1000_10': [MRCNN_train_time_1000_10],
    #         'MRCNN_1000_20': [MRCNN_train_time_1000_20],
    #         'MRCNN_3000_4': [MRCNN_train_time_3000_4],
    #         'MRCNN_3000_10': [MRCNN_train_time_3000_10],
    #         'MRCNN_3000_20': [MRCNN_train_time_3000_20],
    #         'MRCNN_4000_4': [MRCNN_train_time_4000_4],
    #         'MRCNN_4000_10': [MRCNN_train_time_4000_10],
    #         'MRCNN_4000_20': [MRCNN_train_time_4000_20]
    #     })
    # CGS_time_1000_4 = Test.cal_training_time_1DCGS(BA_1000_4, BA_1000_4_label)
    # CGS_time_1000_10 = Test.cal_training_time_1DCGS(BA_1000_10, BA_1000_10_label)
    # CGS_time_1000_20 = Test.cal_training_time_1DCGS(BA_1000_20, BA_1000_20_label)
    # CGS_time_3000_4 = Test.cal_training_time_1DCGS(BA_3000_4, BA_3000_4_label)
    # CGS_time_3000_10 = Test.cal_training_time_1DCGS(BA_3000_10, BA_3000_10_label)
    # CGS_time_3000_20 = Test.cal_training_time_1DCGS(BA_3000_20, BA_3000_20_label)
    # CGS_time_4000_4 = Test.cal_training_time_1DCGS(BA_4000_4, BA_4000_4_label)
    # CGS_time_4000_10 = Test.cal_training_time_1DCGS(BA_4000_10, BA_4000_10_label)
    # CGS_time_4000_20 = Test.cal_training_time_1DCGS(BA_4000_20, BA_4000_20_label)
    
    # CGS_time_train_pd = pd.DataFrame({
    #     '1DCGS_1000_4': [CGS_time_1000_4],
    #     '1DCGS_1000_10': [CGS_time_1000_10],
    #     '1DCGS_1000_20': [CGS_time_1000_20],
    #     '1DCGS_3000_4': [CGS_time_3000_4],
    #     '1DCGS_3000_10': [CGS_time_3000_10],
    #     '1DCGS_3000_20': [CGS_time_3000_20],
    #     '1DCGS_4000_4': [CGS_time_4000_4],
    #     '1DCGS_4000_10': [CGS_time_4000_10],
    #     '1DCGS_4000_20': [CGS_time_4000_20]
    # })
    
    
    # # === Assign RCNN training time variables ===
    # RCNN_1000_4 = RCNN_train_time_1000_4
    # RCNN_1000_10 = RCNN_train_time_1000_10
    # RCNN_1000_20 = RCNN_train_time_1000_20
    # RCNN_3000_4 = RCNN_train_time_3000_4
    # RCNN_3000_10 = RCNN_train_time_3000_10
    # RCNN_3000_20 = RCNN_train_time_3000_20
    # RCNN_4000_4 = RCNN_train_time_4000_4
    # RCNN_4000_10 = RCNN_train_time_4000_10
    # RCNN_4000_20 = RCNN_train_time_4000_20
    
    # # === Assign MRCNN training time variables ===
    # MRCNN_1000_4 = MRCNN_train_time_1000_4
    # MRCNN_1000_10 = MRCNN_train_time_1000_10
    # MRCNN_1000_20 = MRCNN_train_time_1000_20
    # MRCNN_3000_4 = MRCNN_train_time_3000_4
    # MRCNN_3000_10 = MRCNN_train_time_3000_10
    # MRCNN_3000_20 = MRCNN_train_time_3000_20
    # MRCNN_4000_4 = MRCNN_train_time_4000_4
    # MRCNN_4000_10 = MRCNN_train_time_4000_10
    # MRCNN_4000_20 = MRCNN_train_time_4000_20
    
    # # === Assign CGS training time variables ===
    # CGS_1000_4 = CGS_time_1000_4
    # CGS_1000_10 = CGS_time_1000_10
    # CGS_1000_20 = CGS_time_1000_20
    # CGS_3000_4 = CGS_time_3000_4
    # CGS_3000_10 = CGS_time_3000_10
    # CGS_3000_20 = CGS_time_3000_20
    # CGS_4000_4 = CGS_time_4000_4
    # CGS_4000_10 = CGS_time_4000_10
    # CGS_4000_20 = CGS_time_4000_20
    
    # # === Now safe to build RCNN_time_train_pd ===
    # RCNN_time_train_pd = pd.DataFrame({
    #     'RCNN_1000_4': [RCNN_1000_4],
    #     'RCNN_1000_10': [RCNN_1000_10],
    #     'RCNN_1000_20': [RCNN_1000_20],
    #     'RCNN_3000_4': [RCNN_3000_4],
    #     'RCNN_3000_10': [RCNN_3000_10],
    #     'RCNN_3000_20': [RCNN_3000_20],
    #     'RCNN_4000_4': [RCNN_4000_4],
    #     'RCNN_4000_10': [RCNN_4000_10],
    #     'RCNN_4000_20': [RCNN_4000_20]
    # })
    
    # # === RCNN, MRCNN, CGS dictionaries ===
    # RCNN_times = {
    #     '1000_4': RCNN_1000_4,
    #     '1000_10': RCNN_1000_10,
    #     '1000_20': RCNN_1000_20,
    #     '3000_4': RCNN_3000_4,
    #     '3000_10': RCNN_3000_10,
    #     '3000_20': RCNN_3000_20,
    #     '4000_4': RCNN_4000_4,
    #     '4000_10': RCNN_4000_10,
    #     '4000_20': RCNN_4000_20,
    # }
    
    # MRCNN_times = {
    #     '1000_4': MRCNN_1000_4,
    #     '1000_10': MRCNN_1000_10,
    #     '1000_20': MRCNN_1000_20,
    #     '3000_4': MRCNN_3000_4,
    #     '3000_10': MRCNN_3000_10,
    #     '3000_20': MRCNN_3000_20,
    #     '4000_4': MRCNN_4000_4,
    #     '4000_10': MRCNN_4000_10,
    #     '4000_20': MRCNN_4000_20,
    # }
    
    # CGS_times = {
    #     '1000_4': CGS_1000_4,
    #     '1000_10': CGS_1000_10,
    #     '1000_20': CGS_1000_20,
    #     '3000_4': CGS_3000_4,
    #     '3000_10': CGS_3000_10,
    #     '3000_20': CGS_3000_20,
    #     '4000_4': CGS_4000_4,
    #     '4000_10': CGS_4000_10,
    #     '4000_20': CGS_4000_20,
    # }
    
    # # === RT DataFrame ===
    # df = pd.DataFrame({
    #     'Dataset': list(RCNN_times.keys()),
    #     'RCNN_Time': list(RCNN_times.values()),
    #     'MRCNN_Time': list(MRCNN_times.values()),
    #     '1D_CGS_Time': list(CGS_times.values()),
    # })
    
    # print(df)
    # df.to_csv('training_times_comparison2.csv', index=False)
