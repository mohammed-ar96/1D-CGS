# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 01:05:20 2025

@author: لينوفو
"""
import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import os
import community.community_louvain as community
import Test
import Embeddings
import Models
import random
from torch.utils.data import DataLoader,TensorDataset
from tqdm import tqdm
from torch_geometric.data import Data, Batch
sns.set_style('ticks')


# ====================================================m

# Generate subgraph and adjacency matrix
def generate_subgraph(G,node_list):
    """Obtain the adjacent matrix according to the target node
    Parameters:
        G: target network
        Node_list: a list of target nodes and fixed number of neighbor nodes
    Return:
        G_sub: sub -network
        A: The corresponding neighbor network
    """
    G_sub = nx.Graph()
    L = len(node_list) 
    encode = dict(zip(node_list,list(range(L)))) 
    subgraph = nx.subgraph(G,node_list) 
    subgraph_edges = list(subgraph.edges()) 
    new_subgraph_edges = []
    for i,j in subgraph_edges:
        new_subgraph_edges.append((encode[i],encode[j]))
    G_sub.add_edges_from(new_subgraph_edges)
    A = np.zeros([L,L])
    for i in range(L):
        for j in range(L):
            if G_sub.has_edge(i,j) and (i!=j):
                A[i,j]=1
    return G_sub,A

def transform(A,degree_list):
    """Convert according to the rules
    Parameters:
        A: Administrative matrix
        degree_list: The corresponding value of the selected node
    Return:
        B: The embedded matrix of the single channel
    """
    B = A
    B[0,1:] = A[0,1:]*(np.array(degree_list)[1:])
    B[1:,0] = A[1:,0]*(np.array(degree_list)[1:])
    for i in range(len(degree_list)):
        B[i,i]=degree_list[i]
    return B

def setup_seed(seed):
    torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

# Train the hybrid model
def train_1DCGS_model(features, edge_index, labels, num_epochs=3001, lr=0.005):
    """training function for hybrid model"""

    # Matplotlib style
    plt.style.use('seaborn-ticks')
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Arial']

    # Font sizes
    tick_fontsize = 14
    label_fontsize = 16
    title_fontsize = 18
    legend_fontsize = 14

    plt.rc('xtick', labelsize=tick_fontsize)
    plt.rc('ytick', labelsize=tick_fontsize)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Models.HybridModel(feature_dim=features.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    features = features.to(device)
    edge_index = edge_index.to(device)
    labels = labels.to(device)

    loss_history = []
    model.train()

    for epoch in tqdm(range(num_epochs)):
        optimizer.zero_grad()
        pred = model(features, edge_index)
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.15f}")

    # Plot training curve
    plt.figure(figsize=(8, 6))
    plt.plot(loss_history, label='BA_1000_4', color='red', linewidth=2)
    plt.xlabel('Epoch', fontsize=label_fontsize)
    plt.ylabel('Loss', fontsize=label_fontsize)
    plt.title('Training Loss Curve', fontsize=title_fontsize)
    plt.legend(fontsize=legend_fontsize)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    # plt.savefig('training-CGS.pdf', format='pdf', bbox_inches='tight')
    plt.show()

    return model

# ====================================================




def transform1(A,degree_list,com_list,shell_list):
    """Convert according to the rules
    Parameters:
        A: Administrative matrix
        degree_list: The corresponding value of the selected node
        com_list: the number of associations corresponding to the selected node
        shell_list: The K nuclear value corresponding to the selected node
    Return:
        B: The embedded matrix of the 3 channel
    """
    B1 = A.copy()
    B2 = A.copy()
    B3 = A.copy()
    B1[0,1:] = B1[0,1:]*(np.array(degree_list)[1:])
    B1[1:,0] = B1[1:,0]*(np.array(degree_list)[1:])

    B2[0,1:] = B2[0,1:]*(np.array(com_list)[1:])
    B2[1:,0] = B2[1:,0]*(np.array(com_list)[1:])
    
    B3[0,1:] = B3[0,1:]*(np.array(shell_list)[1:])
    B3[1:,0] = B3[1:,0]*(np.array(shell_list)[1:])

    for i in range(len(degree_list)):
        B1[i,i]=degree_list[i]
        B2[i,i]=com_list[i]
        B3[i,i]=shell_list[i]
    
    B = torch.zeros(3,A.shape[0],A.shape[0])
    B[0,:,:]= torch.from_numpy(B1).float()
    B[1,:,:]= torch.from_numpy(B2).float()
    B[2,:,:]= torch.from_numpy(B3).float()
    return B

def transform2(A,degree_list,com_list):
    """Convert according to the rules
    Parameters:
        A: Administrative matrix
        degree_list: The corresponding value of the selected node
        com_list: the number of associations corresponding to the selected node
        shell_list: The K nuclear value corresponding to the selected node
    Return:
        B: The embedded matrix of the 3 channel
    """
    B1 = A.copy()
    B2 = A.copy()
    B1[0,1:] = B1[0,1:]*(np.array(degree_list)[1:])
    B1[1:,0] = B1[1:,0]*(np.array(degree_list)[1:])

    B2[0,1:] = B2[0,1:]*(np.array(com_list)[1:])
    B2[1:,0] = B2[1:,0]*(np.array(com_list)[1:])
    

    for i in range(len(degree_list)):
        B1[i,i]=degree_list[i]
        B2[i,i]=com_list[i]
    
    B = torch.zeros(2,A.shape[0],A.shape[0])
    B[0,:,:]= torch.from_numpy(B1).float()
    B[1,:,:]= torch.from_numpy(B2).float()
    return B

def neighbor_degree(G):
    """Neighbors: The degree of the neighbor node"""
    nodes = list(G.nodes())
    degree = dict(G.degree())
    n_degree = {}
    for node in nodes:
        nd = 0
        neighbors = G.adj[node]
        for nei in neighbors:
            nd+=degree[nei]
        n_degree[node]=nd
    return n_degree

def Louvain(G):
    """Use the Louvain algorithm to divide the community"""
    def com_number(G,partition,community_dic):
        """Get the number of communities connected by each node and the size of the community """
        com_num = {}
        com_size = {}
        for node in G.nodes():
            com_size[node]=len(community_dic[partition[node]])
            com_set = set([partition[node]])
            for nei in list(G.adj[node]):
                if partition[nei] not in com_set:
                    com_set.add(partition[nei])
            com_num[node]=len(com_set)
        return com_num,com_size
    
    partition = community.best_partition(G)
    community_name = set(list(partition.values()))
    community_dic = {}
    for each in community_name:
        a = []
        for node in list(partition.keys()):
            if partition[node] == each:
                a.append(node)
        community_dic[each] = a
    com_num,com_size = com_number(G,partition,community_dic)
    return community_dic,com_num,com_size


    
def load_graph(path):
    """Read the network according to the side
    Parameters:
        PATH: The path of network storage
    Return:
        G: Reading network
    """
    G = nx.read_edgelist(path,create_using=nx.Graph())
    return G

def load_sir_list(path):
    """
    Read the results of SIR under different BETA situation simulation
    Parameters:
        PATH: The root path to store SIR results
    Return:
        SIR simulation results of each node
    """
    sir_list = []
    for i in range(10):
        sir = pd.read_csv(path+str(i)+'.csv')
        sir_list.append(dict(zip(np.array(sir['Node'],dtype=str),sir['SIR'])))
    return sir_list

# Single -channel DataLoader
def Get_DataLoader(data,label,batch_size,L):
    """Create a single DataLoader
    Parameters:
        Data: Data sets are dictionaries (keys are nodes, and values ​​are matrix)
        Label: Data sets are dictionaries (keys are nodes, and values ​​are matrix)
        BATCH_SIZE: How many training each time
    Return:
        Loader: DataLoader.
    
    """
# First turn all Numpy to TORCH
    torch_set = torch.empty(len(data),1,L,L)
    for inx,matrix in enumerate(data.values()):
        torch_set[inx,:,:,:] = torch.from_numpy(matrix)

    sir_torch = torch.empty(len(label),1)
    for inx,v in enumerate(label.values()):
        sir_torch[inx,:] = v

    # Create DataLoader
    deal_data = TensorDataset(torch_set,sir_torch)
    Loader = DataLoader(dataset=deal_data,batch_size=batch_size,shuffle=True)
    return Loader

# Three -channel DataLoader
def Get_DataLoader1(data,label,batch_size,L):
    """Create three -channel DataLoader
    Parameters:
        Data: Data sets are dictionaries (keys are nodes, and values ​​are matrix)
        Label: Data sets are dictionaries (keys are nodes, and values ​​are matrix)
        BATCH_SIZE: How many training each time
    Return:
        Loader: DataLoader.
    
    """
    # First turn all Numpy to TORCH
    torch_set = torch.empty(len(data),3,L,L)
    for inx,matrix in enumerate(data.values()):
        torch_set[inx,:,:,:] = matrix

    sir_torch = torch.empty(len(label),1)
    for inx,v in enumerate(label.values()):
        sir_torch[inx,:] = v

    # Create DataLoader
    deal_data = TensorDataset(torch_set,sir_torch)
    Loader = DataLoader(dataset=deal_data,batch_size=batch_size,shuffle=True)
    return Loader



def train_model(loader,model,num_epochs,lr,L,path=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss() # Loss function
    optimizer = optim.Adam(model.parameters(),lr=lr) # Optimization function
    loss_list = [] # List of LOSS
    for epoch in tqdm(range(num_epochs)):
        for data,targets in loader:
            data = data.to(device)
            targets = targets.float().to(device)
            model = model.to(device)
            pred = model(data)
            loss = criterion(pred,targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            loss_list.append(np.float64(loss.data))
        if epoch % 100 == 0:
            print("Loss:{}".format(loss.data))
    # # Draw the change of LOSS
    # plt.figure(figsize=(8,6),dpi=100)
    # plt.xlabel('epochs',fontsize=14,fontweight='bold')
    # plt.ylabel('loss',fontsize=14,fontweight='bold')
    # plt.plot(np.arange(0,num_epochs,10),loss_list,marker='o',c='r',label='BA_1000_4_28')
    # plt.legend()
    # plt.title('RCNN Model Training Loss')
    # plt.show()
    # if path:
    #     torch.save(model.state_dict(),path)
        
    return model,loss_list

def calculate(l1,l2):
    p = np.mean((np.array(l2)-np.array(l1))/np.array(l1))
    return p


def normalization(data):
    data_norm = (data-data.min(axis=0))/(data.max(axis=0)-data.min(axis=0))
    return data_norm


def save_csv(rcnn,mrcnn,dc,ks,nd,bc,path):
    data = pd.DataFrame({'RCNN':rcnn,'MRCNN':mrcnn,'DC':dc,'K-core':ks,'ND':nd,'BC':bc})
    data.to_csv(path,index=False)
    
def extract_subgraph(G,node):
    candidates = set([node])
    for neighbor in list(G.adj[node]):
        candidates.add(neighbor)
    subnetwork = nx.subgraph(G, list(candidates))
    return subnetwork
        


def Vc(G,community):
    vc = {}
    for node in list(G.nodes()):
        com_set = set({community[node]})
        for nei in list(G.adj[node]):
            if community[nei] not in com_set:
                com_set.add(community[nei])
        vc[node] = len(com_set)
    return vc



def SIR(G,infected,beta=0.1,miu=1):
    """Sir Model
    Input:
        G: Original network
        Infected: The infected node
        MIU: The probability of recovery
    Return:
        Re: After simulation N times, the average infection scale of the node
    
    """
    N = 1000
    re = 0
    
    while N > 0:
        inf = set(infected) # The initial infected node collection
        R = set() # Recovery node
        while len(inf) != 0:
            newInf = []
            for i in inf:
                for j in G.neighbors(i):
                    k = random.uniform(0,1)
                    if (k < beta) and (j not in inf) and (j not in R):
                        newInf.append(j)
                k2 = random.uniform(0, 1)
                if k2 >miu:
                    newInf.append(i)
                else:
                    R.add(i)
            inf = set(newInf)
        re += len(R)+len(inf)
        N -= 1
    return re/1000.0

def SIR_dict(G,beta=0.1,miu=1,real_beta=None,a=1.5):
    """Get the SIR result of all nodes of the entire network
    Input:
        G: target network
        Beta: Probability of Communication
        MIU: Restore probability
        Real_beta: Probability of communication calculated by formula
    Return:
        SIR_DIC: Dictionary of all node communication capabilities
    """
    
    node_list = list(G.nodes())
    SIR_dic = {}
    if real_beta:
        dc_list = np.array(list(dict(G.degree()).values()))
    
        beta = a*(float(dc_list.mean())/(float((dc_list**2).mean())-float(dc_list.mean())))
    print('beta:',beta)
    for node in tqdm(node_list):
        sir = SIR(G,infected=[node],beta=beta,miu=miu)
        SIR_dic[node] = sir
    return SIR_dic

def save_sir_dict(dic,path):
    """The result of storing SIR
    Parameters:
        DIC: SIR results (DICT)
        PATH: target storage path
    """
    node = list(dic.keys())
    sir = list(dic.values())
    Sir = pd.DataFrame({'Node':node,'SIR':sir})
    Sir.to_csv(path,index=False)

def SIR_betas(G,a_list,root_path):
    """SIR in different beta conditions
    Parameters:
        G: target network
        A_List: The list of transmission probability is a list of how many times the transmission threshold
    """
    sir_list = []
    for inx,a in enumerate(a_list):
        sir_dict = SIR_dict(G,real_beta=True,a=a)
        sir_list.append(sir_dict)
        path = root_path+str(inx)+'.csv'
        save_sir_dict(sir_dict,path)

    return sir_list
