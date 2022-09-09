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
M=5
P = 2
etanl = 0.001 
eta_z = 1e-4
N_init = 2
sigma_noise = 1e-2
lamda_n = 0.005


np.random.seed(0)

def randbin(M,N,P):  
    return np.random.choice([0, 1], size=(M,N), p=[P, 1-P])

z_true = pickle.load(open("results/A_wAs_10_fun_3_n.txt","rb"))
N,T = z_true.shape
m_data = randbin(N,T,0.1)


z_noise = np.random.randn(N,T)*sigma_noise
z_noisy = z_true + z_noise
z_tilde_data = np.multiply(z_noisy,m_data) 

def multiprocess_train_lost_list(hyperparam_nu):    


    z_data = pickle.load(open("results/A_wAs_10_fun_3_n.txt","rb"))

    #pdb.set_trace()
    #z_data = z_data[:,0:100] #pickle.load(open("results/A_wAs.txt","rb"))
    ##########################################################################################

    cost,cost_test,A_n,cost_Val,z,NMSE_z_train,NMSE_z_test,NMSE_z_val = learn_model_init(NE, etanl ,lamda_n,P, M,N_init,m_data,z_tilde_data,hyperparam_nu, eta_z,z_true )

    #cost_linear,cost_test_linear,A_l,cost_val_l = learn_model_linear(NE, z_data, A,etal, lamda_l) 
    
    ##########################################################################################

    #pickle.dump(cost,open("lambda_sweep_f_n/cost_"+str(lamda_n)+"_.txt","wb"))
    #pickle.dump(cost_val,open("lambda_sweep_f_n/cost_val_"+str(lamda)+"_.txt","wb"))
    #pickle.dump(cost_test,open("lambda_sweep_f_n/cost_test_"+str(lamda_n)+"_.txt","wb"))
    #pickle.dump(cost_val[NE-1],open("lambda_sweep_f_n/val_lambda_"+str(lamda)+"_.txt","wb"))
    pickle.dump(A_n,open("lambda_sweep_f_n/A_n_"+str(hyperparam_nu)+"_.txt","wb"))
    pickle.dump(z,open("lambda_sweep_f_n/z_data_"+str(hyperparam_nu)+"_.txt","wb"))
    
    pickle.dump(NMSE_z_train,open("lambda_sweep_f_n/NMSE_z_train_"+str(hyperparam_nu)+"_.txt","wb"))
    pickle.dump(NMSE_z_test,open("lambda_sweep_f_n/NMSE_z_test_"+str(hyperparam_nu)+"_.txt","wb"))
    pickle.dump(NMSE_z_train[NE-1],open("lambda_sweep_f_n/NMSE_z_val_"+str(hyperparam_nu)+"_.txt","wb"))


processes = []



hyperparam_nu  = np.arange(0.1,10,0.5)


pdb.set_trace()

pickle.dump(hyperparam_nu,open("lambda_sweep_f_n/lam_LVAR.txt","wb"))
pickle.dump(NE,open("lambda_sweep_f_n/NE.txt","wb"))

    
    
if __name__ ==  '__main__':
    

    for i2 in range(len(hyperparam_nu)):

        p = multiprocessing.Process(target = multiprocess_train_lost_list,args = [np.round(hyperparam_nu[i2],5)])            
        p.start()

        processes.append(p)


    for p1 in processes:
        p1.join()
