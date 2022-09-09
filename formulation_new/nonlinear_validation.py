import numpy as np
import pdb
import pickle
import os
import csv
import multiprocessing
from learn_model import learn_model_balta

from learn_model import learn_model_init

from learn_model import learn_model_genie

M=10

P = 2
NE = 1000
etanl = 0.001 

N_init = 1
np.random.seed(0)

def multiprocess_train_lost_list(lamda):    

    input_data_filename = "data/synthetic/synthetic_dataset_P2_T10000.pickle"
    z_data = pickle.load(open(input_data_filename,"rb"))
    dict = pickle.load(open("function_para/linear_para_dict.pickle","rb"))
    #pdb.set_trace()
    #z_data = z_data[:,0:100] #pickle.load(open("results/A_wAs.txt","rb"))
    ##########################################################################################

    cost,cost_test,A_n,cost_val = learn_model_genie(NE, etanl ,z_data,lamda,P, M,N_init,dict) 
    
    ##########################################################################################

    pickle.dump(cost,           open(results_folder_name+"/cost_"+str(lamda)+"_.txt","wb"))
    pickle.dump(cost_val,       open(results_folder_name+"/cost_val_"+str(lamda)+"_.txt","wb"))
    pickle.dump(cost_test,      open(results_folder_name+"/cost_test_"+str(lamda)+"_.txt","wb"))
    pickle.dump(cost_val[NE-1], open(results_folder_name+"/val_lambda_"+str(lamda)+"_.txt","wb"))
    pickle.dump(A_n,            open(results_folder_name+"/A_n_"+str(lamda)+"_.txt","wb"))





# lam1 = np.arange(0.00001,0.0001,0.00001) 
# lam2 = np.arange(0.0001,0.001,0.0001)
# lam3 = np.arange(0.001,0.01,0.001) 

# lama = np.append(lam1,lam2)
# lam = np.append(lama,lam3)

# 

lam = np.arange(0,1,1)
#lam = np.arange(1,2,1)

results_folder_name = "lambda_sweep_f_n"
if not os.path.isdir(results_folder_name):
    os.mkdir(results_folder_name)

pdb.set_trace()

pickle.dump(lam, open(results_folder_name + "/lam_LVAR.txt","wb"))
pickle.dump(NE,  open(results_folder_name + "/NE.txt","wb"))




if __name__ ==  '__main__':
    
    processes = []
    for i2 in range(len(lam)):

        p = multiprocessing.Process(target = multiprocess_train_lost_list,args = [np.round(lam[i2],5)])            
        p.start()

        processes.append(p)


    for p1 in processes:
        p1.join()
