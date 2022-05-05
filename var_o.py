import sys
from cvxpy import lambda_min
from torch import lu
sys.path.append('code_compare')
import numpy as np
import networkx as nx
#from LinearVAR import scaleCoefsUntilStable
from generating import  nonlinear_VAR_realization
import matplotlib.pyplot as plt
#from NonlinearVAR import NonlinearVAR
from LinearVAR_Kevin import learn_model as learn_model_linear
import networkx as nx
from networkx.generators.random_graphs import erdos_renyi_graph
from learn_model import learn_model
from learn_model import learn_model_init
import pdb
import pickle
import csv

NE = 100
N=10
M=10

P = 2
NE = 5
etanl = 0.01 

N_init = 2

NE = 50
sigma_noise = 0.00001

def randbin(M,N,P):  
    return np.random.choice([0, 1], size=(M,N), p=[P, 1-P])

z_true = pickle.load(open("results/A_wAs_10_fun_3_n.txt","rb"))
N,T = z_true.shape
m_data = randbin(N,T,0.05)


z_noise = np.random.randn(N,T)*sigma_noise
z_noisy = z_true + z_noise
z_tilde_data = np.multiply(z_noisy,m_data)           # masked true data 

z_tilde_data = z_tilde_data[:,0:200]

lamda_n = 0.0025
pdb.set_trace()



def var():

    
    #z_data = pickle.load(open("results/A_wAs_10_fun_3_n.txt","rb"))
    #z_data = pickle.load(open("lundin_2000_n.txt","rb"))


    #pdb.set_trace()
    
    ##########################################################################################

    cost,cost_test,A_n,cost_Val = learn_model_init(NE, etanl ,lamda_n,P, M,N_init,m_data,z_tilde_data)

    #cost_linear,cost_test_linear,A_l,cost_val_l = learn_model_linear(NE, z_data, A,etal, lamda_l) 
    
    ##########################################################################################


    
def main():
    import cProfile
    import pstats

    with cProfile.Profile() as pr:
        var()
    stats = pstats.Stats(pr)
    #stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats(filename='needs_profiling_4.prof')

if __name__ == "__main__":
    main()
