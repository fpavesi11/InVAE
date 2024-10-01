#%%
import os
import torch
import numpy as np
from tqdm import tqdm
import argparse
import random
import igraph

if os.getcwd().split('\\')[-1] != 'metricLearningApproach':
    os.chdir('metricLearningApproach')
    
from DAGgen import *
from DVAE import *
    
g_list_train, g_list_test, graph_args = load_ENAS_graphs('final_structures6', n_types=6, fmt='string', rand_seed=0, with_y=True, burn_in=1000)

print('Number of training graphs:', len(g_list_train))
print('Number of test graphs:', len(g_list_test))
print('Number of node types:', graph_args.num_vertex_type)
print('Maximum number of nodes:', graph_args.max_n)
print('Start vertex type:', graph_args.START_TYPE)
print('End vertex type:', graph_args.END_TYPE)

adj, features = extract_adj_and_features(g_list_train)


#%%
from torch_geometric.data import Data

train_data = [Data(x=features[i], edge_index=adj[i].nonzero().t().contiguous()) for i in range(len(features))]

#%%

