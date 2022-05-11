#!/usr/bin/env python
# coding: utf-8

import sys
import networkx as nx
import pickle
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def payoff_calc_frist_period(G, NR1, x, c):
    payoff = 0

    # adding node reward
    for node_id in G.nodes:
        if x[node_id] == 1:
            payoff = payoff + NR1[node_id]

    # adding agreement bonus
    for edge in G.edges:
        node_id1 = min(edge)
        node_id2 = max(edge)
        if x[node_id1] == x[node_id2]:
            payoff = payoff + c

    return payoff


def payoff_calc_second_period(G, NR1, NR2, x, c):
    payoff = 0

    ### model setup
    m = gp.Model('ntwkOpt')

    ### Create variables
    # action for nodes @t=2
    acts = {}
    for node in G.nodes:
        acts[node] = m.addVar(vtype=GRB.BINARY, name=str(node)+'-act')
    
    # disagreement indicator on acts between neighboring nodes @t=2
    edges_diff = {} 
    for edge in G.edges:
        node1 = min(edge)
        node2 = max(edge) 
        edges_diff[(node1, node2)] = m.addVar(vtype=GRB.BINARY, name=str(node1)+'_'+str(node2)+'-diff')

    ### Set objective
    # (node reward @t=2) + (edge interaction @t=2) + (temporal interaction)
    m.setObjective(
        gp.quicksum([NR2[node]*acts[node] for node in G.nodes]) + 
        gp.quicksum([c*(1-edges_diff[(min(edge), max(edge))]) for edge in G.edges]) + 
        gp.quicksum([c*acts[node] if x[node]==1 else c*(1-acts[node]) for node in G.nodes]),
        GRB.MAXIMIZE
    )

    ### Add constraint: 
    # edge disagreement @t=2
    for edge in G.edges:
        node1 = min(edge)
        node2 = max(edge) 
        m.addConstr(
            edges_diff[(node1, node2)] >= acts[node1] - acts[node2], 
            name='si+_'+str(node1)+'_'+str(node2)
        )
        m.addConstr(
            edges_diff[(node1, node2)] >= acts[node2] - acts[node1], 
            name='si-_'+str(node1)+'_'+str(node2)
        )
    
    ### Optimize model
    m.Params.LogToConsole = 0
    m.optimize()

    # return optimal objective value
    return m.objVal



if __name__ == "__main__":
    # graph size
    size = int(sys.argv[1])

    # instance id
    i = int(sys.argv[2])

    # list of costs
    cs = [0.1, 0.2, 0.3, 0.4, 0.5]

    # list of locality parameters 
    Ls = [0, 2, 4, 6, -1]

    # loading graph
    with open('graphs/graph_size%d.pickle' % (size), 'rb') as f:
        G = pickle.load(f)

    # diameter of graph
    diameter = nx.diameter(G)

    # loading instances 
    with open('instances/instance_NR1s_size%d.pickle' % (size), 'rb') as f:
        NR1s = pickle.load(f)
    NR1 = NR1s[i]

    with open('instances/instance_NR2s_est_size%d.pickle' % (size), 'rb') as f:
        NR2s = pickle.load(f)

    # compute payoff and store to dataframe
    cols = ['c', 'L', 'instance', 'realization', 'payoff']
    nrow = len(cs) * len(Ls) * len(NR2s)
    df = pd.DataFrame(columns=cols, index=range(nrow))
    idx = 0
    for c in cs:
        for L in Ls:
            print('c=%.3f  L=%2d' % (c, L), flush=True)

            # load solution
            with open('solns/solution_size%d_c%.3f_L%d_i%d.pickle' % (size, c, L, i), 'rb') as f:
                x = pickle.load(f)

            # first period payoff
            payoff_1 = payoff_calc_frist_period(G, NR1, x, c)

            # estimated second-period payoff
            for s, NR2 in enumerate(NR2s):
                payoff_2 = payoff_calc_second_period(G, NR1, NR2, x, c)
            
                # update dataframe
                df.loc[idx].instance = i+1
                df.loc[idx].c = c
                if L == -1:
                    df.loc[idx].L = diameter
                else:
                    df.loc[idx].L = L
                df.loc[idx].realization = s
                df.loc[idx].payoff = payoff_1 + payoff_2

                # update index number
                idx = idx + 1 

    # export result
    df.to_csv('payoffs/payoffs_size%d_i%d.csv' % (size, i), index = False)
