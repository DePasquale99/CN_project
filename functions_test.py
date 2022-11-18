import networkx as nx
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import math
from time import time


import functions as fc

#Tests for the D function

def test_distance_norm():
    N_x, N_y, N_z = 10, 10, 10

    lattice = nx.Graph()
    node_counter = 0
    for i in range(N_x): 
      for j in range(N_y): 
        for k in range(N_z):
          lattice.add_node(node_counter, position = np.array([i/N_x, j/N_y, k/N_z]))
          node_counter+=1
    t = time()
    D = fc.nodes_distances(lattice) 
    t = time() - t

    print('execution time = ')
    print(t)

    assert(math.isclose(np.sum(D), 1))
    return

#test for the connectivity rule

def test_NN_norm():
    G = nx.erdos_renyi_graph(1000, 0.5)
    
    t = time()
    K = fc.common_neighbors(G)

    t = time() - t

    print('execution time = ')
    print(t)


    assert(math.isclose(np.sum(K), 1))
    return

#test for the probability dist.

def test_probability_dist():
    N_x, N_y, N_z = 10, 10, 10
    number_links = 200

    lattice = nx.Graph()
    node_counter = 0
    for i in range(N_x): 
      for j in range(N_y): 
        for k in range(N_z):
            lattice.add_node(node_counter, position = np.array([i/N_x, j/N_y, k/N_z]))
            node_counter+=1

    

    for a in range(number_links):
        i = rnd.randint(0, N_x*N_y*N_z)
        j = rnd.randint(0, N_x*N_y*N_z)
        lattice.add_edge(i, j)
    
    P = fc.probability_dist(lattice, -0.98, 0.40, fc.common_neighbors, fc.nodes_distances)

    assert(math.isclose(np.sum(P), 1))
    return

#test_distance_norm()
#test_NN_norm()
