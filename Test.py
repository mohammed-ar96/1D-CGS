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
import RT
import pandas as pd
import time




#conversion TORCH for RCNN 
def to_torch1(data,L):
    """Convert data to TORCH format
    Parameters:
        data: Main function generated data (dictionary format)
        L: Number of target nodes+neighbor nodes
    Return:
        torch_data: Torch format
    """
    torch_data = torch.empty(len(data),1,L,L)
    for inx,matrix in enumerate(data.values()):
        torch_data[inx,:,:] = torch.from_numpy(matrix)
    return torch_data


# M-RCNN conversion TORCH
def to_torch2(data,L):
    """Convert data to TORCH format
    Parameters:
        data: Main function generated data (dictionary format)
        L: Number of target nodes+neighbor nodes
    Return:
        torch_data: Torch format
    """
    torch_data = torch.empty(len(data),3,L,L)
    for inx,matrix in enumerate(data.values()):
        torch_data[inx,:,:] = matrix
    return torch_data

def nodesRank(rank):
    """Helper function to compute node ranks."""
    SR = sorted(rank)
    re = []
    for i in SR:
        re.append(rank.index(i))
    return re

# this function used for calculating the kendall Tau 
def compare_tau(G, model, sir_list, community,ccc,MRCNN, path,path2):
    """Evaluation function"""
    L1=28
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        
        # Get predictions
        features, edge_index, node_to_idx=Embeddings.main2(G)
        features = features.to(device)
        edge_index = edge_index.to(device)
        pred_scores = model(features, edge_index).cpu().numpy()
        
        MRCNN =MRCNN.to(device)
        nodes = list(G.nodes())
        pred_dict = {node: pred_scores[node_to_idx[node]] for node in nodes}
        
        
        
        # Centrality measures
        dc = dict(nx.degree_centrality(G))  # Degree centrality
        ks = dict(nx.core_number(G))        # K-shell
        bc =  nx.betweenness_centrality(G)    # BC centrality
        nd = Utils.neighbor_degree(G)       # Neighbor degree
        vc = Utils.Vc(G, community)         # Community-based centrality
        
        hh,MD=alg2.mixedDegreeDecomposition(G, 0.7, None)
        MDD=dict(sorted(MD.items()))               # Mixed degree decomp.
        hi=alg2.calcHIndexValues(G,3)
        
        
        CGS_pred = [i for i,j in sorted(pred_dict.items(),key=lambda x:x[1],reverse=True)]
        CGS_rank = np.array(nodesRank(CGS_pred),dtype=float)
        
        ccc = ccc.to(device)
        nodes = list(G.nodes())
        torch.manual_seed(5)
        mrcnn_data = to_torch2(Embeddings.main1(G,L1,community),L1).to(device)
        mrcnn_pred = [i for i,j in sorted(dict(zip(nodes,MRCNN(mrcnn_data).to('cpu'))).items(),key=lambda x:x[1],reverse=True)] 
        mrcnn_rank = np.array(nodesRank(mrcnn_pred),dtype=float)
        
        torch.manual_seed(5)
        torch.cuda.empty_cache()
        rcnn_data = to_torch1(Embeddings.main(G,L1),L1).to(device)
        
        
        rcnn_pred = [i for i,j in sorted(dict(zip(nodes,ccc(rcnn_data).to('cpu'))).items(),key=lambda x:x[1],reverse=True)]
        
        rcnn_rank = np.array(nodesRank(rcnn_pred),dtype=float)
        
        
        
        dc_rank = np.array(nodesRank([i for i,j in sorted(dc.items(),key=lambda x:x[1],reverse=True)]),dtype=float)
        ks_rank = np.array(nodesRank([i for i,j in sorted(ks.items(),key=lambda x:x[1],reverse=True)]),dtype=float)
        bc_rank = np.array(nodesRank([i for i,j in sorted(bc.items(),key=lambda x:x[1],reverse=True)]),dtype=float)
        nd_rank = np.array(nodesRank([i for i,j in sorted(nd.items(),key=lambda x:x[1],reverse=True)]),dtype=float)
        vc_rank = np.array(nodesRank([i for i,j in sorted(vc.items(),key=lambda x:x[1],reverse=True)]),dtype=float)
        
        mdd_rank = np.array(nodesRank([i for i, j in sorted(MDD.items(), key=lambda x: x[1], reverse=True)]), dtype=float)
        hi_rank = np.array(nodesRank([i for i, j in sorted(hi.items(), key=lambda x: x[1], reverse=True)]), dtype=float)
        # ksif_rank = np.array(nodesRank([i for i, j in sorted(KSIF.items(), key=lambda x: x[1], reverse=True)]), dtype=float)
        
        CGS_tau_list = []
        dc_tau_list = []
        ks_tau_list = []
        nd_tau_list = []
        bc_tau_list = []
        vc_tau_list = []
        RCNN_tau_list = []
        MRCNN_tau_list = []
        mdd_tau_list = []
        # ksif_tau_list = []
        hi_tau_list = []

        for sir in sir_list:
            sir_sort = [i for i,j in sorted(sir.items(),key=lambda x:x[1],reverse=True)]
            sir_rank = np.array(nodesRank(sir_sort),dtype=float)
            tau1,_ = stats.kendalltau(CGS_rank,sir_rank)
            tau3,_ = stats.kendalltau(dc_rank,sir_rank)
            tau4,_ = stats.kendalltau(ks_rank,sir_rank)
            tau5,_ = stats.kendalltau(nd_rank,sir_rank)
            tau6,_ = stats.kendalltau(bc_rank,sir_rank)
            tau7,_ = stats.kendalltau(vc_rank,sir_rank)
            
            tau8,_ = stats.kendalltau(rcnn_rank,sir_rank)
            
            tau9,_ = stats.kendalltau(mdd_rank,sir_rank)
            tau10,_ = stats.kendalltau(hi_rank,sir_rank)
            tau11,_ = stats.kendalltau(mrcnn_rank,sir_rank)

            CGS_tau_list.append(tau1)
            dc_tau_list.append(tau3)
            ks_tau_list.append(tau4)
            nd_tau_list.append(tau5)
            bc_tau_list.append(tau6)
            vc_tau_list.append(tau7)
            
            RCNN_tau_list.append(tau8)
            mdd_tau_list.append(tau9)
            hi_tau_list.append(tau10)
            MRCNN_tau_list.append(tau11)
            
        a = np.arange(1,2,0.1)
        # outf = open(path+"kendall_CGS.dat", "w")
        # for i in range(10):
        #     outf.write(str(a[i]) + " " + str(CGS_tau_list[i]) + "\n")
        # outf.close()
        
        # outf = open(path+"kendall_MRCNN.dat", "w")
        # for i in range(10):
        #     outf.write(str(a[i]) + " " + str(MRCNN_tau_list[i]) + "\n")
        # outf.close()
        
        # outf = open(path+"kendall_RCNN.dat", "w")
        # for i in range(10):
        #     outf.write(str(a[i]) + " " + str(RCNN_tau_list[i]) + "\n")
        # outf.close()
        
        # outf = open(path+"kendall_DC.dat", "w")
        # for i in range(10):
        #     outf.write(str(a[i]) + " " + str(dc_tau_list[i]) + "\n")
        # outf.close()
        
        # outf = open(path+"kendall_Kcore.dat", "w")
        # for i in range(10):
        #     outf.write(str(a[i]) + " " + str(ks_tau_list[i]) + "\n")
        # outf.close()
        
        # outf = open(path+"kendall_BC.dat", "w")
        # for i in range(10):
        #     outf.write(str(a[i]) + " " + str(bc_tau_list[i]) + "\n")
        # outf.close()
        
        # outf = open(path+"kendall_VC.dat", "w")
        # for i in range(10):
        #     outf.write(str(a[i]) + " " + str(vc_tau_list[i]) + "\n")
        # outf.close()
        
        # outf = open(path+"kendall_MDD.dat", "w")
        # for i in range(10):
        #     outf.write(str(a[i]) + " " + str(mdd_tau_list[i]) + "\n")
        # outf.close()
            
        # outf = open(path+"kendall_HI.dat", "w")
        # for i in range(10):
        #     outf.write(str(a[i]) + " " + str(hi_tau_list[i]) + "\n")
        # outf.close()
            
        # outf = open(path+"kendall_ND.dat", "w")
        # for i in range(10):
        #     outf.write(str(a[i]) + " " + str(nd_tau_list[i]) + "\n")
        # outf.close()
        
        
            
        return CGS_tau_list,RCNN_tau_list,dc_tau_list,ks_tau_list,nd_tau_list,bc_tau_list,vc_tau_list,mdd_tau_list,hi_tau_list,MRCNN_tau_list


def calculate_similarity(G, SIR_dic, community, CGS, MRCNN):
    L = 28
    nodes = list(G.nodes())
    
    
    features, edge_index, node_to_idx = Embeddings.main2(G)
    cgs_pred = dict(zip(nodes, CGS(features, edge_index)))
    
    mrcnn_data = to_torch2(Embeddings.main1(G, L, community), L)
    mrcnn_pred = dict(zip(nodes, MRCNN(mrcnn_data)))
    
    CGS_tau_list = []
    MRCNN_tau_list = []
    SIR_list = []
    for node in nodes:
        CGS_tau_list.append(cgs_pred[node].detach().numpy())  # Detach and convert to numpy
        MRCNN_tau_list.append(mrcnn_pred[node].detach().numpy())  # Detach and convert to numpy
        SIR_list.append(SIR_dic[node])
    
    result_pd = pd.DataFrame({
        'SIR': SIR_list,
        '1D-CGS': CGS_tau_list,
        'MRCNN': MRCNN_tau_list  
    })
    return result_pd


# this function used to store the actual vitality of each node and then used for Jaccarad similarity
def calc_ranks(G, model, sir_list, community,ccc,MRCNN, path,path2):
    """Evaluation function"""
    L1=28
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        
        # Get predictions
        features, edge_index, node_to_idx=Embeddings.main2(G)
        features = features.to(device)
        edge_index = edge_index.to(device)
        pred_scores = model(features, edge_index).cpu().numpy()
        
        MRCNN =MRCNN.to(device)
        nodes = list(G.nodes())
        pred_dict = {node: pred_scores[node_to_idx[node]] for node in nodes}
        
        
        
        # Centrality measures
        dc = dict(nx.degree_centrality(G))  # Degree centrality
        ks = dict(nx.core_number(G))        # K-shell
        bc =  nx.betweenness_centrality(G)    # BC centrality
        nd = Utils.neighbor_degree(G)       # Neighbor degree
        vc = Utils.Vc(G, community)         # Community-based centrality
        hh,MD=alg2.mixedDegreeDecomposition(G, 0.7, None)
        MDD=dict(sorted(MD.items()))               # Mixed degree decomp.
        hi=alg2.calcHIndexValues(G,3)
        
        
        CGS_pred = [i for i,j in sorted(pred_dict.items(),key=lambda x:x[1],reverse=True)]
        CGS_rank = np.array((CGS_pred),dtype=float)
        
        ccc = ccc.to(device)
        nodes = list(G.nodes())
        torch.manual_seed(5)
        mrcnn_data = to_torch2(Embeddings.main1(G,L1,community),L1).to(device)
        mrcnn_pred = [i for i,j in sorted(dict(zip(nodes,MRCNN(mrcnn_data).to('cpu'))).items(),key=lambda x:x[1],reverse=True)] 
        mrcnn_rank = np.array((mrcnn_pred),dtype=float)
        
        torch.manual_seed(5)
        torch.cuda.empty_cache()
        rcnn_data = to_torch1(Embeddings.main(G,L1),L1).to(device)
        
        
        rcnn_pred = [i for i,j in sorted(dict(zip(nodes,ccc(rcnn_data).to('cpu'))).items(),key=lambda x:x[1],reverse=True)]
        
        rcnn_rank = np.array((rcnn_pred),dtype=float)
        
        
        
        dc_rank = np.array(([i for i,j in sorted(dc.items(),key=lambda x:x[1],reverse=True)]),dtype=float)
        ks_rank = np.array(([i for i,j in sorted(ks.items(),key=lambda x:x[1],reverse=True)]),dtype=float)
        bc_rank = np.array(([i for i,j in sorted(bc.items(),key=lambda x:x[1],reverse=True)]),dtype=float)
        nd_rank = np.array(([i for i,j in sorted(nd.items(),key=lambda x:x[1],reverse=True)]),dtype=float)
        vc_rank = np.array(([i for i,j in sorted(vc.items(),key=lambda x:x[1],reverse=True)]),dtype=float)
        
        mdd_rank = np.array(([i for i, j in sorted(MDD.items(), key=lambda x: x[1], reverse=True)]), dtype=float)
        hi_rank = np.array(([i for i, j in sorted(hi.items(), key=lambda x: x[1], reverse=True)]), dtype=float)
            
        a = np.arange(1,2,0.1)
        
        # jacc======================
        # outf = open(path2 + "jacc_CGS.dat", "w")        
        # for i in range(len(nodes)):
        #     outf.write(str((CGS_rank[i])) + "\n")        
        # outf.close()
        
        # outf = open(path2+"jacc_RCNN.dat", "w")
        # for i in range(len(nodes)):
        #     outf.write(str((rcnn_rank[i])) + "\n")        
        # outf.close()
        
        # outf = open(path2+"jacc_MRCNN.dat", "w")
        # for i in range(len(nodes)):
        #     outf.write(str((mrcnn_rank[i])) + "\n")        
        # outf.close()
        
        # outf = open(path2+"jacc_DC.dat", "w")
        # for i in range(len(nodes)):
        #     outf.write(str((dc_rank[i])) + "\n")        
        # outf.close()
        
        # outf = open(path2+"jacc_HI.dat", "w")
        # for i in range(len(nodes)):
        #     outf.write(str((hi_rank[i])) + "\n")        
        # outf.close()
        
        # outf = open(path2+"jacc_Kcore.dat", "w")
        # for i in range(len(nodes)):
        #     outf.write(str((ks_rank[i])) + "\n")        
        # outf.close()
        
        # outf = open(path2+"jacc_VC.dat", "w")
        # for i in range(len(nodes)):
        #     outf.write(str((vc_rank[i])) + "\n")        
        # outf.close()
        
        # outf = open(path2+"jacc_ND.dat", "w")
        # for i in range(len(nodes)):
        #     outf.write(str((nd_rank[i])) + "\n")        
        # outf.close()
        
        # outf = open(path2+"jacc_MDD.dat", "w")
        # for i in range(len(nodes)):
        #     outf.write(str((mdd_rank[i])) + "\n")        
        # outf.close()
        
        # outf = open(path2+"jacc_BC.dat", "w")
        # for i in range(len(nodes)):
        #     outf.write(str((bc_rank[i])) + "\n")        
        # outf.close()
            
        return 0


# just for calc node score, to be used then for MI
def calc_score(G, model, sir_list, community,ccc,MRCNN, path,path2):
    L1=28
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        
        # Get predictions
        nodes = list(G.nodes())
        features, edge_index, node_to_idx=Embeddings.main2(G)
        features = features.to(device)
        edge_index = edge_index.to(device)
        pred_scores = model(features, edge_index).cpu().numpy()
        # 1D-CGS
        pred_dict = {node: pred_scores[node_to_idx[node]] for node in nodes}
        CGS_pred = [j for i,j in sorted(pred_dict.items(),key=lambda x:x[1],reverse=True)]
        CGS_rank = np.array((CGS_pred),dtype=float)
        
        # Centrality measures
        dc = dict(nx.degree_centrality(G))  # Degree centrality
        ks = dict(nx.core_number(G))        # K-shell
        bc = nx.betweenness_centrality(G)    # BC centrality
        nd = Utils.neighbor_degree(G)       # Neighbor degree
        vc = Utils.Vc(G, community)         # Community-based centrality
        hh,MD=alg2.mixedDegreeDecomposition(G, 0.7, None)
        MDD=dict(sorted(MD.items()))               # Mixed degree decomp.
        hi=alg2.calcHIndexValues(G,3)
        
        

        # RCNN
        ccc = ccc.to(device)
        nodes = list(G.nodes())
        torch.manual_seed(5)
        torch.cuda.empty_cache()
        rcnn_data = to_torch1(Embeddings.main(G,L1),L1).to(device)
        rcnn_pred = [float(j.item()) for i, j in sorted(dict(zip(nodes, ccc(rcnn_data).to('cpu'))).items(), key=lambda x: x[1], reverse=True)]
        rcnn_rank = np.array((rcnn_pred),dtype=float)
        
        # MRCNN
        MRCNN =MRCNN.to(device)
        torch.manual_seed(5)
        torch.cuda.empty_cache() 
        mrcnn_data = to_torch2(Embeddings.main1(G,L1,community),L1).to(device)
        mrcnn_pred = [float(j.item()) for i, j in sorted(dict(zip(nodes, MRCNN(mrcnn_data).to('cpu'))).items(), key=lambda x: x[1], reverse=True)]
        mrcnn_rank = np.array((mrcnn_pred),dtype=float)
        
        
        dc_rank = np.array(([j for i,j in sorted(dc.items(),key=lambda x:x[1],reverse=True)]),dtype=float)
        ks_rank = np.array(([j for i,j in sorted(ks.items(),key=lambda x:x[1],reverse=True)]),dtype=float)
        bc_rank = np.array(([j for i,j in sorted(bc.items(),key=lambda x:x[1],reverse=True)]),dtype=float)
        nd_rank = np.array(([j for i,j in sorted(nd.items(),key=lambda x:x[1],reverse=True)]),dtype=float)
        vc_rank = np.array(([j for i,j in sorted(vc.items(),key=lambda x:x[1],reverse=True)]),dtype=float)
        
        mdd_rank = np.array(([j for i, j in sorted(MDD.items(), key=lambda x: x[1], reverse=True)]), dtype=float)
        hi_rank = np.array(([j for i, j in sorted(hi.items(), key=lambda x: x[1], reverse=True)]), dtype=float)
        # ksif_rank = np.array(([j for i, j in sorted(KSIF.items(), key=lambda x: x[1], reverse=True)]), dtype=float)
        
            
        a = np.arange(1,2,0.1)
        # outf = open(path2 + "MO_CGS.dat", "w")        
        # for i in range(len(nodes)):
        #     outf.write(str((CGS_rank[i])) + "\n")        
        # outf.close()
        
        # outf = open(path2+"MO_RCNN.dat", "w")
        # for i in range(len(nodes)):
        #     outf.write(str((rcnn_rank[i])) + "\n")        
        # outf.close()
        
        
        # outf = open(path2+"MO_MRCNN.dat", "w")
        # for i in range(len(nodes)):
        #     outf.write(str((mrcnn_rank[i])) + "\n")        
        # outf.close()
        
        # outf = open(path2+"MO_DC.dat", "w")
        # for i in range(len(nodes)):
        #     outf.write(str((dc_rank[i])) + "\n")        
        # outf.close()
        
        # outf = open(path2+"MO_ND.dat", "w")
        # for i in range(len(nodes)):
        #     outf.write(str((nd_rank[i])) + "\n")        
        # outf.close()
        
        # outf = open(path2+"MO_Kcore.dat", "w")
        # for i in range(len(nodes)):
        #     outf.write(str((ks_rank[i])) + "\n")        
        # outf.close()
        
        # outf = open(path2+"MO_BC.dat", "w")
        # for i in range(len(nodes)):
        #     outf.write(str((bc_rank[i])) + "\n")        
        # outf.close()
        
        # outf = open(path2+"MO_VC.dat", "w")
        # for i in range(len(nodes)):
        #     outf.write(str((vc_rank[i])) + "\n")        
        # outf.close()
        
        # outf = open(path2+"MO_MDD.dat", "w")
        # for i in range(len(nodes)):
        #     outf.write(str((mdd_rank[i])) + "\n")        
        # outf.close()
            
        # outf = open(path2+"MO_HI.dat", "w")
        # for i in range(len(nodes)):
        #     outf.write(str((hi_rank[i])) + "\n")        
        # outf.close()
            
            
        return 0


# just for calc Runing Time\ stored in algorithm-times.txt
def compare_RT(G, model, sir_list, community,ccc,MRCNN,path):
    L1=28
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
         
        RT.start()
        RT.elapsedTime(path+'\n',os.path.basename(path))

        # Get predictions
        nodes = list(G.nodes())
        RT.start()
        features, edge_index, node_to_idx = Embeddings.main2(G)
        features = features.to(device)
        edge_index = edge_index.to(device)
        pred_scores = model(features, edge_index).cpu().numpy()
        
        # 1D-CGS
        pred_dict = {node: pred_scores[node_to_idx[node]] for node in nodes}
        CGS_pred = [i for i,j in sorted(pred_dict.items(),key=lambda x:x[1],reverse=True)]
        RT.elapsedTime('1D-CGS',os.path.basename(path))
        CGS_rank = np.array((CGS_pred),dtype=float)
        
        # RCNN
        ccc = ccc.to(device)
        nodes = list(G.nodes())
        torch.manual_seed(5)
        torch.cuda.empty_cache()
        RT.start()
        rcnn_data = to_torch1(Embeddings.main(G,L1),L1).to(device)
        rcnn_pred = [i for i,j in sorted(dict(zip(nodes,ccc(rcnn_data).to('cpu'))).items(),key=lambda x:x[1],reverse=True)]
        RT.elapsedTime('RCNN',os.path.basename(path))
        rcnn_rank = np.array((rcnn_pred),dtype=float)
        
        # MRCNN
        MRCNN =MRCNN.to(device)
        torch.manual_seed(5)
        torch.cuda.empty_cache() 
        RT.start()
        mrcnn_data = to_torch2(Embeddings.main1(G,L1,community),L1).to(device)
        mrcnn_pred = [i for i,j in sorted(dict(zip(nodes,MRCNN(mrcnn_data).to('cpu'))).items(),key=lambda x:x[1],reverse=True)] 
        RT.elapsedTime('MRCNN',os.path.basename(path))
        mrcnn_rank = np.array((mrcnn_pred),dtype=float)
        # =====================================================================================
        # Centrality measures
        RT.start()
        bc = nx.betweenness_centrality(G)       # BC
        RT.elapsedTime('BC',os.path.basename(path))
        
        
        RT.start()
        dc = dict(nx.degree_centrality(G))  # Degree centrality
        RT.elapsedTime('DC',os.path.basename(path))
        
        RT.start()
        hi=alg2.calcHIndexValues(G,3)
        RT.elapsedTime('HI',os.path.basename(path))
        
        RT.start()
        ks = dict(nx.core_number(G))        # K-Core
        RT.elapsedTime('KCORE',os.path.basename(path))
        
        RT.start()
        vc = Utils.Vc(G, community)         # Community-based centrality
        RT.elapsedTime('VC',os.path.basename(path))
        
        RT.start()
        hh,MD=alg2.mixedDegreeDecomposition(G, 0.7, None)
        MDD=dict(sorted(MD.items()))               # Mixed degree decomp.
        RT.elapsedTime('MDD',os.path.basename(path))
        
        
        RT.start()
        nd = Utils.neighbor_degree(G)       # Neighbor degree
        RT.elapsedTime('ND',os.path.basename(path))

            
        return 0


# === RCNN ===
def cal_training_time(G, label, L, batch_size, num_epochs, lr):
    start_time = time.time()
    train_data = Embeddings.main(G, L)
    data_loader = Utils.Get_DataLoader(train_data, label, batch_size, L)
    rcnn = Models.CNN0(L)
    RCNN, RCNN_loss = Utils.train_model(data_loader, rcnn, num_epochs, lr, L)
    end_time = time.time()
    print('RCNN',end_time - start_time)
    return end_time - start_time

# === MRCNN ===
def cal_training_time2(G, label, L, batch_size, num_epochs, lr):
    start_time = time.time()
    _,com,_ = Utils.Louvain(G)
    train_data = Embeddings.main1(G, L,com)
    data_loader = Utils.Get_DataLoader1(train_data, label, batch_size, L)
    mrcnn = Models.CNN1(L)
    MRCNN, MRCNN_loss = Utils.train_model(data_loader, mrcnn, num_epochs, lr, L)
    end_time = time.time()
    print('MRCNN',end_time - start_time)
    return end_time - start_time

# === 1D-CGS ===
def cal_training_time_1DCGS(network, labels_dict):
    start_time = time.time()
    features, edge_index, node_to_idx = Embeddings.main2(network)
    nodes = list(network.nodes())
    labels = torch.tensor([labels_dict[node] for node in nodes], dtype=torch.float)
    model = Utils.train_1DCGS_model(features, edge_index, labels)
    end_time = time.time()
    print('1D-CGS=',end_time - start_time)
    return end_time - start_time