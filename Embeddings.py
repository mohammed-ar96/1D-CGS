
import numpy as np
import networkx as nx
import torch
import Utils

# RCNN
def main(G,L):
    data_dict = {}
    node_list = list(G.nodes()) # Get all nodes in the network
    #Each node extract L-1 neighbor node according to the rules
    for node in node_list:
        subset = [node] # Target node+fixed number of neighbor nodes
        one_order = list(G.adj[node]) #First look at the first -order neighbor node
        one_degree = dict(G.degree(one_order)) #Get the degree of the first -order neighbor node
        if len(one_order) >= L-1: #If the first -order neighbor node is enough, then don’t look at the second -order neighbor
            selected_degree = [len(one_order)] # The number of neighbors of the selected node on the original network
            selected_nei = [i for i,j in sorted(one_degree.items(),key=lambda x:x[1],reverse=True)] # Sort the first -order neighbors on a degree
            for nei in selected_nei:
                if (nei not in subset) and (len(subset)<L):
                    subset.append(nei)
                    selected_degree.append(one_degree[nei])
            node_subgraph,node_A = Utils.generate_subgraph(G,subset) # Generate a step matrix
            node_B = Utils.transform(node_A,selected_degree) # Convert
            data_dict[node] = node_B
        
        elif (len(one_order)< L-1) and (len(one_order)!=0): # When the first -order neighbor nodes are not enough and the number of first -order neighbors is 0, find a higher -level neighbor
            selected_degree = [len(one_order)]
            selected_nei = [i for i,j in sorted(one_degree.items(),key=lambda x:x[1],reverse=True)]
            gap = (L-1)-len(selected_nei) # See how much worse
            high_nei = set(selected_nei) # High -level neighbor node
            neis = selected_nei
            count = 0 # Try 50 times, if you exceed 50 times, use padding
            while True:
                if count==50:
                    break
                new_order = set([])
                for nei in neis: # The neighbors traversing each neighbor node
                    nei_nei = list(G.adj[nei])
                    for each in nei_nei:
                        if (each != node) and (each not in high_nei):
                            new_order.add(each)
                new_order_list = list(new_order)
                degree_new = dict(G.degree(new_order_list))
                new_selected_nei = [i for i,j in sorted(degree_new.items(),key=lambda x:x[1],reverse=True)]
                if len(new_selected_nei) >=gap: # Satisfy the quantity
                    for i in range(gap):
                        selected_nei.append(new_selected_nei[i])
            
                    break
                
                elif len(new_selected_nei)<gap: # Not satisfied
                    for new in new_selected_nei:
                        selected_nei.append(new)
                        
                        gap-=1
                    
                    neis = new_order_list
                    for each in neis:
                        high_nei.add(each)
                count+=1

            for neii in selected_nei:
                if neii not in subset:
                    subset.append(neii)
                    selected_degree.append(len(G.adj[neii]))
            padding = L-len(subset)
            node_subgraph,node_A = Utils.generate_subgraph(G,subset)
            node_B = Utils.transform(node_A,selected_degree)
            if padding == 0:
                data_dict[node] = node_B
            else:
                node_B_padding = np.zeros([L,L])
                for row in range(node_B.shape[0]):
                    node_B_padding[row,:node_B.shape[0]] = node_B[row,:]
                data_dict[node] = node_B_padding
        else: #When the node is an isolated node, use a zero matrix of a L*L to indicate
            data_dict[node] = np.zeros([L,L])
    return data_dict

# M-RCNN
def main1(G,L,community):
    """Nodes embedded in generating main programs
    Parameters:
        G: Target network
        L: The size of the embedded matrix (the total number of neighbors including target nodes)
    return:
        data_dict:Store the dictionaries embedded in the matrix each node{v1:matrix_v1,...,v2:matrix_v2,...}
    """
    data_dict = {}
    node_list = list(G.nodes()) # Get all nodes in the network
    #Each node extract L-1 neighbor node according to the rules
    k_shell = dict(nx.core_number(G))
    nd = Utils.neighbor_degree(G)
    for node in node_list:
        subset = [node] #Target node+fixed number of neighbor nodes
        one_order = list(G.adj[node]) #First look at the first -order neighbor node
        one_degree = dict(G.degree(one_order)) #Get the degree of the first -order neighbor node
        if len(one_order) >= L-1: #If the first -order neighbor node is enough, then don’t look at the second -order neighbor
            selected_com = [community[node]] # The number of neighbors of the selected node on the original network
            selected_shell = [k_shell[node]]
            selected_nd = [nd[node]]
            
            selected_nei = [i for i,j in sorted(one_degree.items(),key=lambda x:x[1],reverse=True)] # 按度值对一阶邻居排序
            for nei in selected_nei:
                if (nei not in subset) and (len(subset)<L):
                    subset.append(nei)
                    selected_com.append(community[nei])
                    selected_shell.append(k_shell[nei])
                    selected_nd.append(nd[nei])
            
            node_subgraph,node_A = Utils.generate_subgraph(G,subset) # Generate a step matrix
            node_B = Utils.transform1(node_A,selected_nd,selected_com,selected_shell) # Convert
            data_dict[node] = node_B
        
        elif (len(one_order)< L-1) and (len(one_order)!=0): # When the first -order neighbor nodes are not enough and the number of first -order neighbors is 0, find a higher -level neighbor
            selected_com = [community[node]]
            selected_shell = [k_shell[node]]
            selected_nd = [nd[node]]
            selected_nei = [i for i,j in sorted(one_degree.items(),key=lambda x:x[1],reverse=True)]
            gap = (L-1)-len(selected_nei) # See how much worse
            high_nei = set(selected_nei) # High -level neighbor node
            neis = selected_nei
            count = 0 # Try 50 times, if you exceed 50 times, use padding
            while True:
                if count==50:
                    break
                new_order = set([])
                for nei in neis: # The neighbors traversing each neighbor node
                    nei_nei = list(G.adj[nei])
                    for each in nei_nei:
                        if (each != node) and (each not in high_nei):
                            new_order.add(each)
                new_order_list = list(new_order)
                degree_new = dict(G.degree(new_order_list))
                new_selected_nei = [i for i,j in sorted(degree_new.items(),key=lambda x:x[1],reverse=True)]
                if len(new_selected_nei) >=gap: # Satisfy the quantity
                    for i in range(gap):
                        selected_nei.append(new_selected_nei[i])
                    break
                
                elif len(new_selected_nei)<gap: # Not satisfied
                    for new in new_selected_nei:
                        selected_nei.append(new)
                        gap-=1
                    neis = new_order_list
                    for each in neis:
                        high_nei.add(each)
                count+=1

            for neii in selected_nei:
                if neii not in subset:
                    subset.append(neii)
                    selected_com.append(community[neii])
                    selected_shell.append(k_shell[neii])
                    selected_nd.append(nd[neii])
            padding = L-len(subset)
            node_subgraph,node_A = Utils.generate_subgraph(G,subset)
            node_B = Utils.transform1(node_A,selected_nd,selected_com,selected_shell)
            if padding == 0:
                data_dict[node] = node_B
            else:
                node_B_padding = torch.zeros([3,L,L])
                for row in range(node_B.shape[0]):
                    node_B_padding[:,:node_B.shape[1],:node_B.shape[1]] = node_B
                data_dict[node] = node_B_padding
        else: #When the node is an isolated node, use a zero matrix of a L*L to indicate
            data_dict[node] = torch.zeros([3,L,L])
    return data_dict



# Hybrid Model '1D-CGS'
def main2(G):
    """Prepares data for the hybrid model."""
    data_dict = {}
    node_features = []
    edge_indices = []
    degree_list = []
    node_list = list(G.nodes())
    
    degrees = dict(G.degree())
    neighbor_degrees_sum = Utils.neighbor_degree(G) 
    # ks = nx.core_number(G) 
    # _,r,_ = Utils.Louvain(G)
    # vc = Utils.Vc(G, r) 
    # Build edge_index
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    edge_index = []
    for u, v in G.edges():
        edge_index.append([node_to_idx[u], node_to_idx[v]])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Create node features
    features = []
    for node in node_list:
        degree = degrees[node]
        avg_neighbor_deg = (neighbor_degrees_sum[node] / degree) if degree != 0 else 0

        # eks = ks[node] * degree + sum(ks[v] * degrees[v] for v in G.neighbors(node))
        # eks = ks[node] + sum(ks[v] for v in G.neighbors(node))
        # Coreness-degree ratio (cdr) to highlight high-coreness/low-degree nodes
        # cdr = ks[node] / (degree + 1e-5)  
        # VC=vc[node]
        # features.append([degree, avg_neighbor_deg, eks, cdr])  # 4D features
        features.append([degree, avg_neighbor_deg]) 
    features = torch.tensor(features, dtype=torch.float)
    return features, edge_index, node_to_idx