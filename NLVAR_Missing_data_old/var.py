import sys

from torch import lu
sys.path.append('code_compare')
import numpy as np
import networkx as nx
from LinearVAR import scaleCoefsUntilStable
from generating import  nonlinear_VAR_realization
import matplotlib.pyplot as plt
from NonlinearVAR import NonlinearVAR
from LinearVAR_Kevin import learn_model as learn_model_linear
import networkx as nx
from networkx.generators.random_graphs import erdos_renyi_graph
from learn_model import learn_model
import pdb
import pickle
import os
import csv


p_test = True
os.system("clear")

if p_test:

    

    N=10
    M=10
    T=2000
    P = 2
    
    #z_data = np.random.rand(N, T)


    # pickle.dump(NE,open("results/NE.txt","wb"))
    # pickle.dump(lamda,open("results/lamda.txt","wb"))

    # A_true =  np.random.rand(N, N, P)



    # for p in range(P): #sparse initialization of A_true

    #     g = erdos_renyi_graph(N, 0.15,seed = None, directed= True)  #remove directed after the meeting 
    #     print(p)
    #     A_t = nx.adjacency_matrix(g)
    #     A_true[:,:,p] = A_t.todense()
    
    #pickle.dump(g,open("results/g.txt","wb")) #note that this connection graph different from A matrix graph 
    #because this is the graph for last P.
    
    
    #pickle.dump(A_true,open("results/A_true_10.txt","wb"))
    def f(x,i): 


        a = b[i]
        for m in range(M):
            a = a + alpha[i][m] * sigmoid(w[i][m]*x-k[i][m])
  
        return a

    def sigmoid(l):
        
        return  1/(1+np.exp(-l)) 



    alpha= pickle.load(open("function_para/alpha.txt","rb"))
    w = pickle.load(open("function_para/w.txt","rb"))
    k = pickle.load(open("function_para/k.txt","rb")) 
    b = pickle.load(open("function_para/b.txt","rb"))
    A = pickle.load(open("function_para/A.txt","rb"))

    # pickle.dump(alpha,open("results/alpha.txt","wb"))
    
    # pickle.dump(z_data,open("results/z_data_woAs.txt","wb")) 
   
    # A_true_1 = scaleCoefsUntilStable(A_true, tol = 0.05, b_verbose = False, inPlace=False)

    A_true_1 = pickle.load(open("results/A_true_1_10.txt","rb"))

    z_data =  nonlinear_VAR_realization(A_true_1, T, f)
    
    
    pickle.dump(z_data,open("results/A_wAs_10_fun_2000.txt","wb"))

    pdb.set_trace()

   
    
    
#     z_data = pickle.load(open("results/lundin/lundin_data.txt","rb"))

#     ##########################################################################################

#     #newobject = NonlinearVAR(N,M,P,filename_tosave = 'model.nlv') #this line meant for comparing the codes if b_comparing = True. However it runs in both cases.


#     cost,cost_test,A_n,cost_Val = learn_model(NE, etanl ,z_data, A, alpha, w, k, b,lamda_n)

#     cost_linear,cost_test_linear,A_l,cost_val_l = learn_model_linear(NE, z_data, A,etal, lamda_l) 
    
#     ##########################################################################################
# #    pdb.set_trace()

#     pickle.dump(cost,open("results/lundin/cost_n.txt","wb"))
#     pickle.dump(cost_test,open("results/lundin/cost_n_test.txt","wb"))

#     pickle.dump(cost_linear,open("lundin/lundin/cost_linear.txt","wb"))
#     pickle.dump(cost_test_linear,open("lundin/lundin/cost_linear_test.txt","wb"))

#     pickle.dump(A_n,open("results/lundin/A_n.txt","wb"))
#     pickle.dump(A_l,open("results/lundin/A_l.txt","wb"))

#    pickle.dump(hat_z_t,open("results/hat_z_t.txt","wb"))
    
    
