import networkx as nx
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import random as rnd
from time import time 

import functions as fc

#Generation of a 3D grid as a dummy for network growth
t = time()
N_x, N_y, N_z = 10, 10, 10
dim_x, dim_y, dim_z = 5, 5, 5
number_links= 10

lattice = nx.Graph()
node_counter = 0
for i in range(N_x): 
  for j in range(N_y): 
    for k in range(N_z):

      lattice.add_node(node_counter, pos_x =i/N_x*dim_x , pos_y = j/N_y*dim_y , pos_z =  k*dim_z/N_z )
      node_counter+=1


for a in range(number_links):
  i = rnd.randint(0, N_x*N_y*N_z)
  j = rnd.randint(0, N_x*N_y*N_z)
  lattice.add_edge(i, j)
t=time()-t
print('initialization time =')
print(t)

connected_lattice, execution_time = fc.generate_net(lattice, 10000, -0.98, 0.42, fc.common_neighbors)


plt.hist(execution_time)
plt.show()
plt.scatter(range(len(execution_time)), execution_time)
plt.show()
nx.write_graphml(connected_lattice, '/home/daniele/Scrivania/PythonCode/Codici/ComplexNetworks/Data/lattice_10000.graphml')


#test con 1000 edges
#4299 secondi con tutta la matrice
#2207 secondi con upper triangular