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


N=10
M=10
T=10000
P = 2

A_true =  np.random.rand(N, N, P)
for p in range(P): #sparse initialization of A_true

    g = erdos_renyi_graph(N, 0.15,seed = None, directed= True)  #remove directed after the meeting 
    print(p)
    A_t = nx.adjacency_matrix(g)
    A_true[:,:,p] = A_t.todense()

#pdb.set_trace()
def identity(x, i):
    return x

os.chdir(sys.path[0]) # no need to use cd command to terminal
model_location = "synthetic/function_para/"
alpha= pickle.load(open(model_location+"alpha.pickle","rb"))
w = pickle.load(open(model_location+"w.pickle","rb"))
k = pickle.load(open(model_location+"k.pickle","rb")) 
b = pickle.load(open(model_location+"b.pickle","rb"))

A_true_1 = scaleCoefsUntilStable(A_true, tol = 0.05, b_verbose = False, inPlace=False)

A_true_1 = pickle.load(open("synthetic/A_true_P_2_T3000.pickle","rb"))  # overwriting previous A_true for comparison purpose
#A_true_1 = A_true_1 *10

#pickle.dump(A_true_1,open("synthetic/A_true_P_2_T3000.pickle","wb"))


### DATA GENERATION ###

#y_data_linear = nonlinear_VAR_realization(A_true_1, T, identity)
z_data,y_data =  nonlinear_VAR_realization(A_true_1, T, f_param, alpha, w, k, b)
#z_data = pickle.load(open("../results/A_wAs_10_fun_3_sliced.txt","rb"))
#z_data = pickle.load(open("../results/A_wAs_10_fun_3_n.txt","rb"))

pickle.dump(z_data, open("synthetic/z_data_genie_10000.pickle","wb"))
######lundin data load into array format from csv ##################
#z_data = genfromtxt('real/lundin.csv', delimiter=',')

#########################################################
z_data_normalized = np.zeros(z_data.shape)

for i in range(N):
    z_data_normalized[i,:] = (z_data[i,:] - np.mean(z_data[i,:]))/np.std(z_data[i,:])
    #z_data_normalized[i,:] = z_data_normalized[i,:]/np.max(z_data[i,:])
#z_data_normalized = z_data_normalized/10

pickle.dump(z_data_normalized, open("synthetic/z_data_genie_10000_normalised.pickle","wb"))

#pickle.dump(z_data_normalized, open("real/lundin_normalised.pickle","wb"))

#z_data_normalized = pickle.load(open("real/lundin_2000_n.pickle","rb"))


#z_data = z_data/10
### DATA VISUALIZATION ###
#z_data_normalized  = pickle.load(open("synthetic/synthetic_dataset_1.pickle","rb"))

#

rc('axes', linewidth=2)
figure, axis = plt.subplots(2)
legend_prop = {'weight':'bold'}
#axis.plot((y_data[0,:],z_data[0,:]),"bo")
axis[0].plot(z_data[0,:])
axis[1].plot(y_data[0,:])

axis[0].set_title("Data generation Non Linear VAR")
axis[0].set_xlabel("t",fontsize=20)
axis[0].set_ylabel("z",fontsize=20)
axis[1].set_xlabel("t",fontsize=20)
axis[1].set_ylabel("y",fontsize=20)
for index in range(1):
    axis[index].grid()
    axis[index].legend(prop={"size":20})
    axis[index].xaxis.set_tick_params(labelsize=20)
    axis[index].yaxis.set_tick_params(labelsize=20)


figure, axis = plt.subplots(1)
legend_prop = {'weight':'bold'}
axis.plot(y_data[0,:],z_data_normalized[0,:],"bo")


axis.set_title("Data generation Non Linear VAR")
axis.set_xlabel("y",fontsize=20)
axis.set_ylabel("z",fontsize=20)
axis.grid()
axis.legend(prop={"size":20})
axis.xaxis.set_tick_params(labelsize=20)
axis.yaxis.set_tick_params(labelsize=20)

### to visualise linear data
#pdb.set_trace()
# figure, axis = plt.subplots(1)

# legend_prop = {'weight':'bold'}
# axis.plot(np.transpose(y_data_linear),linewidth=3)
# #axis.set_ylim([0, 1])
# axis.set_title("Cost cmparison LinearVAR vs Non LinearVAR")
# axis.set_xlabel("Epoch",fontsize=20)
# axis.set_ylabel("NMSE",fontsize=20)
# axis.grid()
# axis.legend(prop={"size":20})
# axis.xaxis.set_tick_params(labelsize=20)
# axis.yaxis.set_tick_params(labelsize=20)

#pickle.dump(z_data,open("results/A_wAs_10_fun_3_sliced.pickle","wb"))

plt.show()