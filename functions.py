import networkx as nx
from networkx.algorithms.shortest_paths.generic import shortest_path_length
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import random as rnd
from time import time


def common_neighbors(Graph):
  #associates a probability based on the percentage of common neighbors
  n = len(Graph.nodes())
  K = np.zeros((n, n))
  for i in range(1, n):
    for j in range(i):
      common_neighbors = len(list(nx.common_neighbors(Graph, i, j)))
      if common_neighbors==0:
          K[i][j] = 0
      else:
            total_neighbors = len(list(nx.neighbors(Graph, i))) + len(list(nx.neighbors(Graph, i))) - len(list(nx.common_neighbors(Graph, i, j)))
            K[i][j] = common_neighbors / total_neighbors
  
  K += 1e-6
  K = K/np.sum(K)
  return K


def nodes_distances(Graph): 
  #takes the graph and calcualtes the distance matrix between nodes i,j
  n = len(Graph.nodes())
  D = np.zeros((n, n))
  for i in range(1, n):
    for j in range(i):
      difference =np.array([Graph.nodes[i]['x']-Graph.nodes[j]['x'], Graph.nodes[i]['y']-Graph.nodes[j]['y'], Graph.nodes[i]['z']-Graph.nodes[j]['z']])
      D[i][j] = np.sqrt(np.dot(difference,difference))

  D = D/np.sum(D)
  return D


def probability_dist(Graph, alpha, beta, connectivity_rule, distance):
  #calculates the normalized matrix for edge generation, according to the formula p = E^alpha*K^beta
  A = nx.to_numpy_array(Graph)
  n = len(A)
  P = np.zeros((n,n))
  K = connectivity_rule(Graph)
  for i in range(1, n):
        for j in range(i):
          if (A[i][j] == 0):
            P[i][j] = distance[i][j]**alpha*K[i][j]**beta 
  return P



def generate_net(Graph, edges_goal, alpha, beta, connectivity_rule):
  #goal of this function is to run the generative algorithm, takes the sparse net, the target number of edges, the generative rule and it's parameters.
  G0 = Graph
  nodes_number = len(Graph.nodes())
  distance = nodes_distances(Graph)
  t0 = time()
  tc = 0
  T_vec = np.array([])
  while len(Graph.edges())<edges_goal :
    probability = probability_dist(Graph, alpha, beta, connectivity_rule, distance)
    normalized_probability = probability/np.sum(probability)
    extraction = np.random.rand(nodes_number,nodes_number)
    for i in range(1, nodes_number):
      for j in range(i):
        if normalized_probability[i][j] > extraction[i][j]:
          Graph.add_edge(i, j)
    
    t = time()-t0 - np.sum(T_vec)
    T_vec = np.append(T_vec, t)

    print('Number of edges =')
    print(len(Graph.edges()))
    print('seconds needed =')
    print(t)

  print('total time = ')
  print(np.sum(T_vec))
  return Graph, T_vec

def print_degree(Graph):
  #prints degree distribution of the graph
    n = len(Graph.nodes())


    degrees = list((d for n, d in Graph.degree()))
    nbins = max(i for i in degrees)

    plt.hist(degrees, nbins)

    plt.show()

def print_clustering(Graph):
  #prints clustering coefficient distribution for the graph
    n = len(Graph.nodes())


    clustering = list((nx.clustering(Graph)[str(c)] for c in range(n)))
    
    plt.hist(clustering)
    plt.show()


def print_edge_length(Graph):
  #prints the distribution of edge lenght of the graph
    n = len(Graph.nodes())

    edge_list = list(Graph.edges())
    lenght_list = np.array([])
    for i, j in edge_list:
        difference = np.array([Graph.nodes[i]['x']-Graph.nodes[j]['x'], Graph.nodes[i]['y']-Graph.nodes[j]['y'], Graph.nodes[i]['z']-Graph.nodes[j]['z']])
        distance = np.sqrt(np.dot(difference, difference))
        np.append(lenght_list, distance)

    plt.hist(lenght_list)
    plt.show()


def print_betweenness_centrality(Graph):
  #prints the betweenness centrality distribution
    n = len(Graph.nodes())

    betweenness_centrality = list((nx.betweenness_centrality(Graph)[str(i)] for i in range(n)))
    plt.hist(betweenness_centrality)
    plt.show()






def net_analysis(Graph):
  
  print_degree(Graph)
  print_clustering(Graph)
  print_edge_length(Graph)
  print_betweenness_centrality(Graph)

  return







