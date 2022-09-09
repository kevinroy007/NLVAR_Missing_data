import numpy as np
import pdb
import pickle
import multiprocessing

from learn_model import learn_model_linear

import os
os.system("clear")

NE = 200
N=24
M=10
P = 2
#  
etal = 0.001
A = np.ones((N,N,P))

def multiprocess_train_lost_list(lamda):    

    input_data_filename = "data/real/lundin_normalised.pickle"
    z_data = pickle.load(open(input_data_filename,"rb"))


    ##########################################################################################

    cost_linear,cost_test_linear,A_l,cost_val  = learn_model_linear(NE, z_data, A,etal, lamda) 
    
    ##########################################################################################

    pickle.dump(cost_linear,         open(results_folder_name + "/cost_linear_"+str(lamda)+"_.txt","wb"))
    pickle.dump(cost_val,            open(results_folder_name + "/cost_linear_val_"+str(lamda)+"_.txt","wb"))
    pickle.dump(cost_test_linear,    open(results_folder_name + "/cost_linear_test_"+str(lamda)+"_.txt","wb"))
    pickle.dump(cost_val[NE-1],      open(results_folder_name + "/val_lambda_"+str(lamda)+"_.txt","wb"))
    pickle.dump(A_l,                 open(results_folder_name + "/A_l_"+str(lamda)+"_.txt","wb"))





lam = np.arange(0.0001,0.01,0.0002)
lam = np.arange(1,2,1)

results_folder_name = "lambda_sweep_f"
if not os.path.isdir(results_folder_name):
    os.mkdir(results_folder_name)

#pdb.set_trace()

pickle.dump(lam, open(results_folder_name + "/lam_LVAR.txt","wb"))
pickle.dump(NE,  open(results_folder_name + "/NE.txt","wb"))

    
if __name__ ==  '__main__':
     
    processes = []
    for i2 in range(len(lam)):

        p = multiprocessing.Process(target = multiprocess_train_lost_list,args = [np.round(lam[i2],4)])            
        p.start()

        processes.append(p)


    for p1 in processes:
        p1.join()
