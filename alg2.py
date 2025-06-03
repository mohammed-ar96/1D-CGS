# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 17:05:34 2024

@author: لينوفو
"""
import networkx as nx
import numpy as np
from scipy import stats
from collections import Counter
from scipy.sparse import identity, csr_matrix
from scipy.sparse.linalg import inv


# references: Li H, Shang Q, Deng Y. A generalized gravity model for influential spreaders identification in complex networks[J].
# Chaos, Solitons & Fractals, 2021, 143: 110456.
def cal_SP(G):
    SP = {}
    for i in G.nodes():
        ci = nx.clustering(G, i)
        SP[i] = np.exp(-2.0*ci)*G.degree(i)
    return SP

# ================================
# H-index from Identifying influential nodes in social networks: A voting approach
def calcH0IndexValues(nxG):
    result = [nxG.degree(v) for v in nx.nodes(nxG)]
    return result



def count_h_index(h_list):

    h_list = sorted(h_list, reverse=True)
    _len = len(h_list)
    i = 0
    while (i < _len and h_list[i] > i):
        i += 1
    return i


def cal_h_index(G, n, h_neg_dic):
    assert n >= 0, 'n>=0' 

    if n == 0:
        h_index_dic = {}  
        for n_i in nx.nodes(G):
            h_index_dic[n_i] = nx.degree(G, n_i)
        return h_index_dic
    else:
        h_index_dic = {}
        n = n - 1
        h0_index_dic = cal_h_index(G, n, h_neg_dic)
        # print(n,h0_index_dic)
        for n_i in nx.nodes(G):
            h_list = []
            for neg in h_neg_dic[n_i]:
                h_list.append(h0_index_dic[neg])
            h_index_dic[n_i] = count_h_index(h_list)
        return h_index_dic


def calcHIndexValues(nxG, n):  
    h_neg_dic = {}
    for n_i in nx.nodes(nxG):
        a = []
        for neg in nx.neighbors(nxG, n_i):
            a.append(neg)
        h_neg_dic[n_i] = a
    result_dic = cal_h_index(nxG, n, h_neg_dic)
    
    return result_dic


def mixedDegreeDecomposition(Graph, Lambda, withWeight):
    '''
    mixed degree decomposition
    see https://www.sciencedirect.com/science/article/abs/pii/S0375960113002260
    When λ = 0, the MDD method coincides with the k-shell method. 
    When λ = 1, the MDD method is equivalent to the degree centrality method.
    -----------------
    Graph: directed networkx graph with edgeWeight
    Lambda: tenable parameter between 0 and 1
    withWeight: if True then consider weight
    '''
    
    graphToRemove = Graph.to_undirected()
    
    shellRank = dict()
    MDDRank = dict()
    degreeAll = dict(graphToRemove.degree)

    kShell = 0
    
    while len(graphToRemove.nodes) > 0:
        shellRank[kShell] = []
        minKmValue = 0
        while minKmValue <= kShell:
            minKmValue = len(Graph.nodes)
            for node in graphToRemove.nodes:
                
                kmValue = graphToRemove.degree(node) + Lambda*(degreeAll[node] - graphToRemove.degree(node))
                # compute kmValue
                if kmValue < minKmValue:
                    minKmValue = kmValue
                    # update minKmValue
                
                if kmValue <= kShell:
                    # check for nodes to remove
                    shellRank[kShell].append(node)
                    MDDRank[node] = kmValue
        
            graphToRemove.remove_nodes_from(shellRank[kShell])
            # remove nodes
        kShell += 1
    
    return shellRank, MDDRank

# def kShellIterationFactor(Graph):
#     '''
#     k shell iteration factor
#     see https://www.sciencedirect.com/science/article/abs/pii/S0378437116302333
#     -------------------------
#     Graph: networkx graph without edgeWeight
#     '''
#     shellRank = dict()
#     KSiFRank = dict()
#     graphToRemove = Graph.to_undirected()
#     coreNumber = nx.core_number(graphToRemove)
#     # core number
    
#     kShell = 0
    
#     # do until next shell level
#     while len(graphToRemove.nodes) > 0:
#         minDegree = 0
#         mTurn = 0
#         shellRank[kShell] = dict()
#         # do kshell
        
#         while minDegree <= kShell:
#             minDegree = len(Graph.nodes)
#             shellRank[kShell][mTurn] = []
#             # do iteration
#             for node in graphToRemove.nodes:
#                 if graphToRemove.degree(node) <= kShell:
#                     # check for nodes to remove
#                     shellRank[kShell][mTurn].append(node)
                    
#                 if graphToRemove.degree(node) < minDegree:
#                     # update minimum degree
#                     minDegree = graphToRemove.degree(node) 
#                 KSiFRank[node] = mTurn
            
#             graphToRemove.remove_nodes_from(shellRank[kShell][mTurn]) # remove nodes
#             mTurn += 1
            
#         for nturn in range(mTurn):
#             for node in shellRank[kShell][nturn]:
#                 KSiFRank[node] = (1 + KSiFRank[node]/mTurn) * coreNumber[node]
        
#         kShell += 1
    
#     return shellRank, KSiFRank

