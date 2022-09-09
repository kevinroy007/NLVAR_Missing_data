import sys
sys.path.append(sys.path[0]+'/..')
import numpy as np
import networkx as nx
from LinearVAR import scaleCoefsUntilStable
from generating import  nonlinear_VAR_realization
import matplotlib.pyplot as plt
import networkx as nx
from networkx.generators.random_graphs import erdos_renyi_graph
import pickle
import os
from matplotlib import rc
import pdb
from basic_functions import f_param
from generating import linear_VAR_realization
os.system("clear")
np.random.seed(0)
import csv
from numpy import genfromtxt


z_data = genfromtxt('real/lundin.csv', delimiter=',')
N,T = z_data.shape

#########################################################
z_data_normalized = np.zeros(z_data.shape)

for i in range(N):
    z_data_normalized[i,:] = (z_data[i,:] - np.mean(z_data[i,:]))/np.std(z_data[i,:])
    #z_data_normalized[i,:] = z_data_normalized[i,:]/np.max(z_data[i,:])
#z_data_normalized = z_data_normalized/10

pickle.dump(z_data_normalized, open("real/lundin_normalised.pickle","wb"))


