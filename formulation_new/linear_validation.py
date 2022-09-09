import numpy as np
import pdb
import pickle
import multiprocessing

from learn_model import learn_model_linear

import os
os.system("clear")


NE = 20
N=10
M=10 
P = 2
etal = 0.0001

A = np.ones((N,N,P))
#np.random.seed(0)
dict = {}  #there is no need of passing dictionary to linear validatoin but needed because linear model calls learn model which requires dictonary
def multiprocess_train_lost_list(lamda):    

    

    input_data_filename = "data/synthetic/synthetic_dataset_P2_T2000.pickle"
    #input_data_filename = "results/z_wAs_10_fun_n_2000.txt"
    z_data = pickle.load(open(input_data_filename,"rb"))

    #pdb.set_trace()
    #z_data = z_data[:,0:100] 
    ##########################################################################################

    cost_linear,cost_test_linear,A_l,cost_val  = learn_model_linear(NE, z_data, A,etal, lamda,dict) 
    
    ##########################################################################################

    pickle.dump(cost_linear,        open("lambda_sweep_f/cost_linear_"+str(lamda)+"_.txt","wb"))
    pickle.dump(cost_val,           open("lambda_sweep_f/cost_linear_val_"+str(lamda)+"_.txt","wb"))
    pickle.dump(cost_test_linear,   open("lambda_sweep_f/cost_linear_test_"+str(lamda)+"_.txt","wb"))
    pickle.dump(cost_val[NE-1],     open("lambda_sweep_f/val_lambda_"+str(lamda)+"_.txt","wb"))
    pickle.dump(A_l,                open("lambda_sweep_f/A_l_"+str(lamda)+"_.txt","wb"))





lam1 = np.arange(0.001,0.01,0.001)  
lam2 = np.arange(0.01,0.3,0.02) 


lam = np.append(lam1,lam2) 

lam = np.arange(1,2,1)

results_folder_name = "lambda_sweep_f"
if not os.path.isdir(results_folder_name):
    os.mkdir(results_folder_name)



pickle.dump(lam,   open(results_folder_name + "/lam_LVAR.txt","wb"))
pickle.dump(NE,    open(results_folder_name  + "/NE.txt","wb"))


    
    
if __name__ ==  '__main__':
    
    processes = []
    for i2 in range(len(lam)):

        p = multiprocessing.Process(target = multiprocess_train_lost_list,args = [np.round(lam[i2],4)])            
        p.start()

        processes.append(p)


    for p1 in processes:
        p1.join()
