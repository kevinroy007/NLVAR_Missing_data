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
NE = 1
etanl = 0.001 
N_init = 1
np.random.seed(0)
sigma_noise = 1e-1
hyperparam_nu = 10
eta_z = 1e-4


input_data_filename = "data/synthetic/synthetic_dataset_P2_T2000.pickle"
z_true = pickle.load(open(input_data_filename,"rb"))
z_true = z_true[:,0:200]
pdb.set_trace()
dict = pickle.load(open("function_para/linear_para_dict.pickle","rb"))



##################  making data noisy and with missing entries ######################
np.random.seed(0)

def randbin(M,N,P):  
    return np.random.choice([0, 1], size=(M,N), p=[P, 1-P])

#z_true = pickle.load(open("results/A_wAs_10_fun_3_n.txt","rb"))
N,T = z_true.shape
missing = 0.05
m_data = randbin(N,T,missing)  # means 5 percent missing data..


z_noise = np.random.randn(N,T)*sigma_noise
z_noisy = z_true + z_noise
z_tilde_data = np.multiply(z_noisy,m_data) 

####################################################################################
def var(lamda):
#pdb.set_trace()

    

    ##########################################################################################

    cost,cost_test,A_n,cost_val,z = learn_model_balta(NE, etanl ,z_tilde_data,lamda,P, M,N_init,dict,m_data,hyperparam_nu,eta_z,z_true) 

    ##########################################################################################

    pickle.dump(cost,           open(results_folder_name+"/cost_"+str(lamda)+"_.txt","wb"))
    pickle.dump(cost_val,       open(results_folder_name+"/cost_val_"+str(lamda)+"_.txt","wb"))
    pickle.dump(cost_test,      open(results_folder_name+"/cost_test_"+str(lamda)+"_.txt","wb"))
    pickle.dump(cost_val[NE-1], open(results_folder_name+"/val_lambda_"+str(lamda)+"_.txt","wb"))
    pickle.dump(A_n,            open(results_folder_name+"/A_n_"+str(lamda)+"_.txt","wb"))

    pickle.dump(z,            open(results_folder_name+"/z_learned"+str(lamda)+"_.txt","wb"))


results_folder_name = "lambda_sweep_f_n_"+str(missing)
if not os.path.isdir(results_folder_name):
    os.mkdir(results_folder_name)

# lam1 = np.arange(0.001,0.01,0.002)
# lam2 = np.arange(0.01,0.1,0.02)
# lam = np.append(lam1,lam2)
lam = np.arange(0.001,0.002,0.001)


pdb.set_trace()



pickle.dump(lam, open(results_folder_name + "/lam_LVAR.txt","wb"))
pickle.dump(NE,  open(results_folder_name + "/NE.txt","wb"))


if __name__ ==  '__main__':
    
    processes = []
    for i2 in range(len(lam)):

        p = multiprocessing.Process(target = var,args = [np.round(lam[i2],7)])            
        p.start()

        processes.append(p)

    for p1 in processes:
        p1.join()