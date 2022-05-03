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
from learn_model import learn_model_init

import pdb
import pickle
import os
import csv
import multiprocessing


os.system("clear")

NE = 100
N=10
M=10

P = 2
NE = 50
etanl = 0.01 

N_init = 2

NE = 50
def randbin(M,N,P):  
    return np.random.choice([0, 1], size=(M,N), p=[P, 1-P])

m_p = randbin(10,1000,0.05)

z_data_real = pickle.load(open("results/A_wAs_10_fun_3_n.txt","rb"))
z_data_mask = np.multiply(z_data_real,m_p)           # masked true data 

N,T = z_data_real.shape()  
z_data = np.random.rand(N,T)                         # the paramerter to be learned

pdb.set_trace()

def multiprocess_train_lost_list(lamda):    


    z_data = pickle.load(open("results/A_wAs_10_fun_3_n.txt","rb"))

    #pdb.set_trace()
    #z_data = z_data[:,0:100] #pickle.load(open("results/A_wAs.txt","rb"))
    ##########################################################################################

    cost,cost_test,A_n,cost_val,z_data,cost_history_missing = learn_model_init(NE, etanl ,z_data,lamda,P, M,N_init,m_p,z_data_mask) 
    
    ##########################################################################################

    pickle.dump(cost,open("lambda_sweep_f_n/cost_"+str(lamda)+"_.txt","wb"))
    pickle.dump(cost_val,open("lambda_sweep_f_n/cost_val_"+str(lamda)+"_.txt","wb"))
    pickle.dump(cost_test,open("lambda_sweep_f_n/cost_test_"+str(lamda)+"_.txt","wb"))
    pickle.dump(cost_val[NE-1],open("lambda_sweep_f_n/val_lambda_"+str(lamda)+"_.txt","wb"))
    pickle.dump(A_n,open("lambda_sweep_f_n/A_n_"+str(lamda)+"_.txt","wb"))

    pickle.dump(z_data,open("lambda_sweep_f_n/z_data_"+str(lamda)+"_.txt","wb"))
    pickle.dump(cost_history_missing,open("lambda_sweep_f_n/cost_history_missing_"+str(lamda)+"_.txt","wb"))

processes = []



lam = np.arange(0.001,0.05,0.002)


pdb.set_trace()

pickle.dump(lam,open("lambda_sweep_f_n/lam_LVAR.txt","wb"))
pickle.dump(NE,open("lambda_sweep_f_n/NE.txt","wb"))

    
    
if __name__ ==  '__main__':
    

    for i2 in range(len(lam)):

        p = multiprocessing.Process(target = multiprocess_train_lost_list,args = [np.round(lam[i2],5)])            
        p.start()

        processes.append(p)


    for p1 in processes:
        p1.join()
