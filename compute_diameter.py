#!/usr/bin/env python
# coding: utf-8

import networkx as nx
import pickle
import sys

if __name__ == "__main__":
	fid = open('graphs/diameter.txt', 'w')
	size = int(sys.argv[1]) 
	with open('graphs/graph_size%d.pickle' % (size), 'rb') as f:
		G = pickle.load(f)
	fid.write('size=%7d  diameter=%2d\n' % (size, nx.diameter(G)))
	fid.close()

