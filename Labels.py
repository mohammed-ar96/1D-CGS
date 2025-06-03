# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 12:05:20 2025

@author: لينوفو
"""
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import os
import Models
import Utils
import Test
import Embeddings
sns.set_style('ticks')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

if __name__ == '__main__':
    # Fixed random seed
    Utils.setup_seed(5)
    # Import the network
    # generate artificial networks
    
    # BA_3000_4 = Utils.load_graph('.//BA_3000_4.txt')

    # BA_3000_4_label = Utils.SIR_dict(BA_3000_4,real_beta=True)
    # BA_3000_4_pd=pd.DataFrame({'Nodes':list(BA_3000_4_label.keys()),'SIR':list(BA_3000_4_label.values())})  
        
    

    
    dataset = Utils.load_graph('./Networks/real/sex.txt')
    
    a_list = np.arange(1.0,2.0,0.1)
    
    _SIR = Utils.SIR_betas(dataset,a_list,'./SIR results/sex/sex_')