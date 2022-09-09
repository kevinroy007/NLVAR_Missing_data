import sys
sys.path.append('code_compare')
import numpy as np
#from LinearVAR import scaleCoefsUntilStable
#from generating import  nonlinear_VAR_realization
#import matplotlib.pyplot as plt
import pdb
from compute_gradients_Linear import compute_gradients as compute_gradients_l

# need to modify yet making 
#data generation remain Nonlinear
#investigate the need for updationg parameters for matrix A
#try to work it out for on your work sheet on ipad

def learn_model(NE,z_data, A_l,eta,lamda): #TODO: make A, alpha, w, k, b optional
    
    N, T = z_data.shape
    N2,N3,P = A_l.shape
    assert N==N2 and N==N3
    # document inputs
    

    
    cost_history = np.zeros(NE)
    cost_history_test = np.zeros(NE)
    cost_history_val = np.zeros(NE)

    for epoch in range(NE):  
        print("Linear epoch",epoch)
        cost = np.zeros(T)
        cost_test = np.zeros(T)
        cost_val = np.zeros(T)
        hat_z_t = np.zeros((N,T))

        for t in range(P, T):   
            
            dC_dA,cost[t],cost_test[t],cost_val[t] = compute_gradients_l(z_data, A_l, t)

            for i1 in range(N):
                for i2 in range(N):
                    for p in range(P):
                        fw = A_l[i1,i2,p] - eta*dC_dA[i1,i2,p]
                        if fw == 0:
                            A_l[i1,i2,p] = 0
                        else:
                            a1 = 1 - eta*lamda/(abs(fw))
                            A_l[i1,i2,p]  = fw * max(0,a1)
        v_denominators = np.sum(np.square(z_data), axis=0)

        cost_history[epoch] = sum(cost)/sum(v_denominators[0:int(0.7*T)])
        cost_history_test[epoch] = sum(cost_test)/sum(v_denominators[int(0.7*T):int(0.9*T)])
        cost_history_val[epoch] = sum(cost_val)/sum(v_denominators[int(0.9*T):-1])



    return cost_history,cost_history_test,A_l,cost_history_val




