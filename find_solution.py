#!/usr/bin/env python
# coding: utf-8

import sys
import networkx as nx
import pickle
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time


def opt_MIP(G, NR1, NR2s, c, focal_node):
    ### number of sets of samples for t=2
    n_sample = len(NR2s)

    ### model setup
    m = gp.Model('ntwkOpt')

    ### Create variables
    # action for nodes @t=1
    acts_1 = {}
    for node in G.nodes:
        acts_1[node] = m.addVar(
            vtype=GRB.BINARY, 
            name=str(node)+'-act-1'
        )
    
    # action for nodes @t=2 for each set 
    acts_2_set = [{} for s in range(n_sample)]
    for s in range(n_sample):
        for node in G.nodes:
            acts_2_set[s][node] = m.addVar(
                vtype=GRB.BINARY, 
                name=str(node)+'-act-2-'+str(s)
            )
    
    # disagreement indicator on acts between t=1 and t=2 for nodes for each set 
    nodes_diff_set = [{} for s in range(n_sample)]
    for s in range(n_sample):
        for node in G.nodes:
            nodes_diff_set[s][node] = m.addVar(
                vtype=GRB.BINARY, 
                name=str(node)+'-diff'
            )
    
    # disagreement indicator on acts between neighboring nodes @t=1
    edges_diff_1 = {}
    for edge in G.edges:
        node1 = min(edge)
        node2 = max(edge) 
        edges_diff_1[(node1, node2)] = m.addVar(
            vtype=GRB.BINARY, 
            name=str(node1)+'_'+str(node2)+'-diff-1'
        )
    
    # disagreement indicator on acts between neighboring nodes @t=2 for each set 
    edges_diff_2_set = [{} for s in range(n_sample)]
    for s in range(n_sample):
        for edge in G.edges:
            node1 = min(edge)
            node2 = max(edge) 
            edges_diff_2_set[s][(node1, node2)] = m.addVar(
                vtype=GRB.BINARY, 
                name=str(node1)+'_'+str(node2)+'-diff-2-'+str(s)
            )

    ### Set objective
    # node reward @t=1 + 
    # edge interaction @t=1 + 
    # 1/n_sample*(
    #   node reward @t=2 for all sets + 
    #   edge interaction @t=2 for all sets + 
    #   temporal interaction for all sets
    # )
    m.setObjective(
        gp.quicksum([NR1[node]*acts_1[node] for node in G.nodes]) + 
        gp.quicksum([c*(1-edges_diff_1[(min(edge), max(edge))]) for edge in G.edges]) + 
        1/n_sample*(
            gp.quicksum([NR2s[s][node]*acts_2_set[s][node] 
                         for node in G.nodes for s in range(n_sample)]) + 
            gp.quicksum([c*(1-edges_diff_2_set[s][(min(edge), max(edge))]) 
                         for edge in G.edges for s in range(n_sample)]) + 
            gp.quicksum([c*(1-nodes_diff_set[s][node])
                         for node in G.nodes for s in range(n_sample)])
        ), 
        GRB.MAXIMIZE
    )

    ### Add constraint: 
    # edge disagreement @t=1
    for edge in G.edges:
        node1 = min(edge)
        node2 = max(edge) 
        m.addConstr(
            edges_diff_1[(node1, node2)] >= acts_1[node1] - acts_1[node2], 
            name='si+_'+str(node1)+'_'+str(node2)+'-1'
        )
        m.addConstr(
            edges_diff_1[(node1, node2)] >= acts_1[node2] - acts_1[node1], 
            name='si-_'+str(node1)+'_'+str(node2)+'-1'
        )
    
    # edge disagreement @t=2 for each set
    for s in range(n_sample):
        for edge in G.edges:
            node1 = min(edge)
            node2 = max(edge) 
            m.addConstr(
                edges_diff_2_set[s][(node1, node2)] >= acts_2_set[s][node1] - acts_2_set[s][node2], 
                name='si+_'+str(node1)+'_'+str(node2)+'-2-'+str(s)
            )
            m.addConstr(
                edges_diff_2_set[s][(node1, node2)] >= acts_2_set[s][node2] - acts_2_set[s][node1], 
                name='si-_'+str(node1)+'_'+str(node2)+'-2-'+str(s)
            )
    
    # temporal disagreement for each set
    for s in range(n_sample):
        for node in G.nodes:
            m.addConstr(
                nodes_diff_set[s][node] >= acts_1[node] - acts_2_set[s][node], 
                name='ti+_'+str(node)+'-'+str(s)
            )
            m.addConstr(
                nodes_diff_set[s][node] >= acts_2_set[s][node] - acts_1[node], 
                name='ti-_'+str(node)+'-'+str(s)
            )
            
    ### Optimize model
    m.Params.LogToConsole = 0
    m.optimize()

    # return solution
    if focal_node == None:
        x = {}
        for v in m.getVars():
            if str.find(v.varName, '-act-1') != -1:
                x[int(str.split(v.varName, '-act-1')[0])] = v.x
        return x
    else:
        for v in m.getVars():
            if v.varName == str(focal_node) + '-act-1':
                return v.x


if __name__ == "__main__":
    # graph size
    size = int(sys.argv[1])

    # agreement bonus [i.e., ferromagnetic]
    c = float(sys.argv[2])

    # locality parameter
    L = int(sys.argv[3])

    # instance id
    i = int(sys.argv[4])

    # loading graph
    with open('graphs/graph_size%d.pickle' % (size), 'rb') as f:
        G = pickle.load(f)

    # loading instances 
    with open('instances/instance_NR1s_size%d.pickle' % (size), 'rb') as f:
        NR1s = pickle.load(f)

    with open('instances/instance_NR2s_sol_size%d.pickle' % (size), 'rb') as f:
        NR2s = pickle.load(f)

    # compute solution
    NR1 = NR1s[i]

    start_time = time.time()
    if L == -1:     
        x = opt_MIP(G, NR1, NR2s, c, None)
    else:
        x = {}
        for num, node_id in enumerate(G.nodes):
            subgraph_G = nx.generators.ego.ego_graph(G, node_id, L)
            x[node_id] = opt_MIP(subgraph_G, NR1, NR2s, c, node_id)

    run_time = time.time() - start_time

    with open('solns/solution_size%d_c%.3f_L%d_i%d.pickle' % (size, c, L, i), 'wb') as f:
        pickle.dump(x, f)
        pickle.dump(run_time, f)
