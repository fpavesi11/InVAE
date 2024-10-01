
import torch
import numpy as np
from tqdm import tqdm
import argparse
import random
import igraph
import warnings



# NOTE: This is a modified version of the original code from D-VAE https://github.com/muhanzhang/D-VAE/blob/master/util.py


# create a parser to save graph arguments
cmd_opt = argparse.ArgumentParser()
graph_args, _ = cmd_opt.parse_known_args()

def load_ENAS_graphs(name, n_types=6, fmt='igraph', rand_seed=0, with_y=True, burn_in=1000):
    # load ENAS format NNs to igraphs or tensors
    g_list = []
    max_n = 0  # maximum number of nodes
    with open('data/%s.txt' % name, 'r') as f:
        for i, row in enumerate(tqdm(f)):
            if i < burn_in:
                continue
            if row is None:
                break
            if with_y:
                row, y = eval(row)
            else:
                row = eval(row)
                y = 0.0
            if fmt == 'igraph':
                g, n = decode_ENAS_to_igraph(row)
            elif fmt == 'string':
                g, n = decode_ENAS_to_tensor(row, n_types)
            max_n = max(max_n, n)
            g_list.append((g, y)) 
    graph_args.num_vertex_type = n_types + 2  # original types + start/end types
    graph_args.max_n = max_n  # maximum number of nodes
    graph_args.START_TYPE = 0  # predefined start vertex type
    graph_args.END_TYPE = 1 # predefined end vertex type
    ng = len(g_list)
    print('# node types: %d' % graph_args.num_vertex_type)
    print('maximum # nodes: %d' % graph_args.max_n)
    random.Random(rand_seed).shuffle(g_list)
    return g_list[:int(ng*0.9)], g_list[int(ng*0.9):], graph_args

def one_hot(idx, length):
    idx = torch.LongTensor([idx]).unsqueeze(0)
    x = torch.zeros((1, length)).scatter_(1, idx, 1)
    return x


def decode_ENAS_to_tensor(row, n_types):
    n_types += 2  # add start_type 0, end_type 1
    if type(row) == str:
        row = eval(row)  # convert string to list of lists
    n = len(row)  # n+2 is the real number of vertices in the DAG
    g = []
    # ignore start vertex
    for i, node in enumerate(row):
        node_type = node[0] + 2  # assign 2, 3, ... to other types
        type_feature = one_hot(node_type, n_types)
        if i == 0:
            edge_feature = torch.zeros(1, n+1)  # a node will have at most n+1 connections
        else:
            edge_feature = torch.cat([torch.FloatTensor(node[1:]).unsqueeze(0), 
                                     torch.zeros(1, n+1-i)], 1)  # pad zeros
        edge_feature[0, i] = 1 # ENAS node always connects from the previous node
        g.append(torch.cat([type_feature, edge_feature], 1))
    # process the output node
    node_type = 1
    type_feature = one_hot(node_type, n_types)
    edge_feature = torch.zeros(1, n+1)
    edge_feature[0, n] = 1  # output node only connects from the final node in ENAS
    g.append(torch.cat([type_feature, edge_feature], 1))
    return torch.cat(g, 0).unsqueeze(0), n+2


def decode_ENAS_to_igraph(row):
    if type(row) == str:
        row = eval(row)  # convert string to list of lists
    n = len(row)
    g = igraph.Graph(directed=True)
    g.add_vertices(n+2)
    g.vs[0]['type'] = 0  # input node
    for i, node in enumerate(row):
        g.vs[i+1]['type'] = node[0] + 2  # assign 2, 3, ... to other types
        g.add_edge(i, i+1)  # always connect from last node
        for j, edge in enumerate(node[1:]):
            if edge == 1:
                g.add_edge(j, i+1)
    g.vs[n+1]['type'] = 1  # output node
    g.add_edge(n, n+1)
    # note that the nodes 0, 1, ... n+1 are in a topological order
    return g, n+2


def flat_ENAS_to_nested(row, n_nodes):
    # transform a flattened ENAS string to a nested list of ints
    if type(row) == str:
        row = [int(x) for x in row.split()]
    cnt = 0
    res = []
    for i in range(1, n_nodes+1):
        res.append(row[cnt:cnt+i])
        cnt += i
        if cnt == len(row):
            break
    return res


def decode_igraph_to_ENAS(g):
    # decode an igraph to a flattend ENAS string
    n = g.vcount()
    res = []
    adjlist = g.get_adjlist(igraph.IN)
    for i in range(1, n-1):
        res.append(int(g.vs[i]['type'])-2)
        row = [0] * (i-1)
        for j in adjlist[i]:
            if j < i-1:
                row[j] = 1
        res += row
    return ' '.join(str(x) for x in res)


"""
import os

if os.getcwd().split('\\')[-1] != 'metricLearningApproach':
    os.chdir('metricLearningApproach')
# Load the ENAS graphs from the file
g_list_train_ten, g_list_test_ten, graph_args_ten = load_ENAS_graphs('final_structures6', n_types=6, fmt='string', rand_seed=0, with_y=True, burn_in=1000)

# Print some information about the graphs
print('Number of training graphs:', len(g_list_train))
print('Number of test graphs:', len(g_list_test))
print('Number of node types:', graph_args.num_vertex_type)
print('Maximum number of nodes:', graph_args.max_n)
print('Start vertex type:', graph_args.START_TYPE)
print('End vertex type:', graph_args.END_TYPE)


selgraph=1
print(g_list_train_ten[selgraph][0][0][:,:8])
print(g_list_train_ten[selgraph][0][0][:,8:])  

example = g_list_train_ten[selgraph][0][0][:,8:].T
example = torch.cat([torch.zeros((example.shape[0], 1)), example], dim=1)
example = torch.cat([example, torch.zeros((1, example.shape[1]))], dim=0)
print(example)



g_list_train, g_list_test, graph_args = load_ENAS_graphs('final_structures6', n_types=6, fmt='igraph', rand_seed=0, with_y=True, burn_in=1000)

# Print some information about the graphs
print('Number of training graphs:', len(g_list_train))
print('Number of test graphs:', len(g_list_test))
print('Number of node types:', graph_args.num_vertex_type)
print('Maximum number of nodes:', graph_args.max_n)
print('Start vertex type:', graph_args.START_TYPE)
print('End vertex type:', graph_args.END_TYPE)



import igraph as ig


# Select a graph from the list
graph = g_list_train[selgraph][0]

# Plot the graph with igraph's plot function
layout = graph.layout_reingold_tilford(mode="in", root=[0])
ig.plot(graph, layout=layout, bbox=(400, 400), margin=20, vertex_label=graph.vs["type"], vertex_size=30, edge_width=1)




g_list_train[0][0].get_adjacency()

"""


#ADDED BY ME: Extract adjacency matrix and features from the graphs
"""
Graph, when extracted as 'string', have and adjacency matrix concatenated with the features of the nodes.
this method extracts the adjacency matrix and the features of the nodes from the graph. As 
adjacency matrix had first column and last row (which are by definition zeros) removed, the method
adds them back. Finally, as node types have the first row excluded (which is always [1,0,0,0,0,0,0,0]) 
we add it back. 
"""

def extract_adj_and_features(g_list):
    warnings.warn('This method works only for final_structures6 dataset')
    adj_list = []
    features_list = []
    for g, _ in tqdm(g_list):
        adj = g[0][:, 8:].T
        adj = torch.cat([torch.zeros((adj.shape[0], 1)), adj], dim=1)
        adj = torch.cat([adj, torch.zeros((1, adj.shape[1]))], dim=0)
        adj_list.append(adj.data)
        features = g[0][:,:8]
        features = torch.cat([torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0.]]), features], dim=0)
        features_list.append(features)
    return adj_list, features_list


