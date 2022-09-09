from operator import ne
import numpy as np
import pickle
import pdb, os, sys

from learn_model import learn_model_genie, learn_model_init, learn_model_linear, learn_model_balta


N=10
M=10
P = 2
etal = 0.001
etanl = 0.001
NE  = 30
lamda_n = 0.001 
N_init = 1

#np.random.seed(0)


file_directory = sys.path[0]
os.chdir(file_directory)

z_data = pickle.load(open("data/synthetic/z_data_genie_10000.pickle","rb"))

model_location = "data/synthetic/function_para/"
alpha= pickle.load(open(model_location+"alpha.pickle","rb"))
w = pickle.load(open(model_location+"w.pickle","rb"))
k = pickle.load(open(model_location+"k.pickle","rb")) 
b = pickle.load(open(model_location+"b.pickle","rb"))

A = pickle.load(open("data/synthetic/A_true_P_2_T3000.pickle","rb")) 

############################################################################

cost,cost_test,A_n,cost_Val = \
    learn_model_genie(NE, etanl ,z_data,lamda_n,P, M,N_init,dict,alpha,w,k,b,A)



#############################################################################

results_folder_name = "genie_results"
if not os.path.isdir(results_folder_name):
    os.mkdir(results_folder_name)

pickle.dump(cost,open(results_folder_name+"/cost_"+str(NE)+"_.txt","wb"))
    
