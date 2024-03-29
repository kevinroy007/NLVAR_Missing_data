import numpy as np
import pdb


def compute_gradients_DEPRECATED(z_data, A, t,onlyForward = False):
     
# COMPUTE_GRADIENTS gets gradients for cost at time t 
# w.r.t. parameters at subnetwork i
# INPUT VARIABLES
# t: integer, is the time instant
# i: integer, is the subnetwork index
# z_data N x T array, containing the given z data
    # N: number of sensors
    # T: number of time instants
    N, T = z_data.shape
# A: N x N x P array, containing VAR parameter matrices
    N,N,P = A.shape
# alpha: N x M array, alpha parameters
    # M: number of units in subnetwork
# w: N x M  array, w parameters
# k: N x M array, k parameters
# b: N array, b parameters
   # gradient of g w.r.t theta     #!LEq(17)
   # Function definitions


    
# Forward pass (function evaluation)
    # (You can use your functions f, g, etc)
    #!LEq(7) from the paper
    tilde_y_tm = np.zeros((N, P+1))   #!LEq(7a) from the paper # here i_prime is used instead of i
    for i_prime in range(N):
        for p in range(1,P+1):
            assert t-p >= 0
            z_i_tmp = z_data[i_prime, t-p]
            tilde_y_tm[i_prime, p] = z_i_tmp

    hat_y_t = np.zeros((N)) #!LEq(7b)
    for i_prime in range(N): 
        for p in range(1,P+1):
            for j in range(N):
                 hat_y_t[i_prime] =  hat_y_t[i_prime] + A[i_prime,j,p-1]*tilde_y_tm[j,p]
  
    
    hat_z_t = np.zeros((N)) #!LEq(7c)
    for i_prime in range(N):    
        hat_z_t[i_prime] = hat_y_t[i_prime]   
    
    #pdb.set_trace()

    # computing cost #!LEq(7d) and vector S #!LEq(8)

    S = 2*( hat_z_t - z_data[:,t])                      
    cost_u = np.sum(np.square(S[:]/2))

    dC_dA = np.zeros((N,N,P)) 
    if (onlyForward==False):    
        for i in range(N):
            for j in range(N):
                for p in range(1,P+1):               
                    dC_dA[i,j,p-1] = S[i]*tilde_y_tm[j, p] 

    return dC_dA,cost_u
    
