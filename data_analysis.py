
import networkx as nx
import functions as fc
import numpy as np

Graph = nx.read_graphml('/home/daniele/Scrivania/PythonCode/Codici/ComplexNetworks/Data/lattice_10000.graphml')
corrupted = True
#momentanely fixes the positions issue
if corrupted == True:
    node_counter = 0
    for i in range(10): 
      for j in range(10): 
        for k in range(10):
           Graph.nodes[str(node_counter)]['x'] = i/2
           Graph.nodes[str(node_counter)]['y'] = j/2
           Graph.nodes[str(node_counter)]['z'] = k/2
           node_counter += 1

fc.print_edge_length(Graph)

#fc.net_analysis(Graph)

