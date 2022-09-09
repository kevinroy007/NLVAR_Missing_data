import sys
#from cvxpy import lambda_sum_largest

from torch import lu
sys.path.append('code_compare')
import numpy as np
import networkx as nx
from LinearVAR import scaleCoefsUntilStable
from generating import  nonlinear_VAR_realization
import matplotlib.pyplot as plt
#from NonlinearVAR import NonlinearVAR
from LinearVAR_Kevin import learn_model as learn_model_linear
import networkx as nx
from networkx.generators.random_graphs import erdos_renyi_graph
from learn_model import learn_model
import pdb
import pickle
import os
import csv
import multiprocessing


os.system("clear")

NE = 200

def multiprocess_train_lost_list(lamda):    

    N=10
    M=10
#    
    P = 2
    NE = 50
#  
    etal = 0.01
   

    A = np.ones((N,N,P))

    
    z_data = pickle.load(open("results/A_wAs_10_fun_3_n.txt","rb"))

    ##########################################################################################

    cost_linear,cost_test_linear,A_l,cost_val  = learn_model_linear(NE, z_data, A,etal, lamda) 
    
    ##########################################################################################

    pickle.dump(cost_linear,open("lambda_sweep_f/cost_linear_"+str(lamda)+"_.txt","wb"))
    pickle.dump(cost_val,open("lambda_sweep_f/cost_linear_val_"+str(lamda)+"_.txt","wb"))
    pickle.dump(cost_test_linear,open("lambda_sweep_f/cost_linear_test_"+str(lamda)+"_.txt","wb"))
    pickle.dump(cost_val[NE-1],open("lambda_sweep_f/val_lambda_"+str(lamda)+"_.txt","wb"))
    pickle.dump(A_l,open("lambda_sweep_f/A_l_"+str(lamda)+"_.txt","wb"))



processes = []

lam = np.arange(0.0001,0.01,0.0002)

#lam = np.arange(1,2,1)

pdb.set_trace()

pickle.dump(lam,open("lambda_sweep_f/lam_LVAR.txt","wb"))
pickle.dump(NE,open("lambda_sweep_f/NE.txt","wb"))


    
    
if __name__ ==  '__main__':
    

    for i2 in range(len(lam)):

        p = multiprocessing.Process(target = multiprocess_train_lost_list,args = [np.round(lam[i2],4)])            
        p.start()

        processes.append(p)


    for p1 in processes:
        p1.join()
