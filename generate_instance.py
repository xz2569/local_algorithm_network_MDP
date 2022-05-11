#!/usr/bin/env python
# coding: utf-8

import sys
import networkx as nx
import pickle
import numpy as np

if __name__ == "__main__":
	# graph size
	size = int(sys.argv[1])

	# degree
	d = 3

	# number of relizations
	n_instance = 10     # number of different NR1
	n_sample_sol = 100 # number of samples of realized NR2 for solving x_1^*
	n_sample_est = 30   # number of samples of realized NR2 for estimating payoff

	# random d-regular graph
	G = nx.generators.random_graphs.random_regular_graph(d, size, seed=123)
	with open('graphs/graph_size%d.pickle' % (size), 'wb') as f:
		pickle.dump(G, f)

	# instances generation 
	# period 1
	NR1s = [{} for i in range(n_instance)]
	for NR1 in NR1s:
		for node_id in G.nodes:
			NR1[node_id] = np.random.uniform(low=-1, high=1) 
	
	with open('instances/instance_NR1s_size%d.pickle' % (size), 'wb') as f:
		pickle.dump(NR1s, f)

	# period 2
	NR2s_sol = [{} for i in range(n_sample_sol)]
	for NR2 in NR2s_sol:
		for node_id in G.nodes:
			NR2[node_id] = np.random.uniform(low=-1, high=1) 
	
	with open('instances/instance_NR2s_sol_size%d.pickle' % (size), 'wb') as f:
		pickle.dump(NR2s_sol, f)

	# period 2
	NR2s_est = [{} for i in range(n_sample_est)]
	for NR2 in NR2s_est:
		for node_id in G.nodes:
			NR2[node_id] = np.random.uniform(low=-1, high=1) 
	
	with open('instances/instance_NR2s_est_size%d.pickle' % (size), 'wb') as f:
		pickle.dump(NR2s_est, f)