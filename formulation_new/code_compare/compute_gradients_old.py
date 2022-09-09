import numpy as np
import pdb, sys
from basic_functions import *
from formulation_B.g_bisection import g_bisection as g_b

def compute_gradients(z_data, A, alpha, w, k, b, t, onlyForward = False):  
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
    N2, M = alpha.shape
    assert(N2 == N)

    
# w: N x M  array, w parameters
# k: N x M array, k parameters
# b: N array, b parameters
   # gradient of g w.r.t theta     #!LEq(17)
   # Function definitions
         
    def f(x,i):
        a3 = alpha[i,:]* sigmoid(w[i,:]*x-k[i,:])
        return a3.sum() +b[i]

    def f_prime(x,i):
        a = alpha[i,:] * sigmoid(w[i,:]*x-k[i,:]) * (1-sigmoid(w[i,:]*x-k[i,:]))*(w[i,:])
        return a.sum()

    def dfalpha(x,i): 
        return sigmoid(w[i,:]*x-k[i,:]) 
   
    def dfb(x,i): 
        return 1
    
    def dfk(x,i): 
        return alpha[i,:] * sigmoid(w[i,:]*x-k[i,:]) * (1-sigmoid(w[i,:]*x-k[i,:]))*(-1)

    def dfw(x,i):         
        return alpha[i,:] * sigmoid(w[i,:]*x-k[i,:]) * (1-sigmoid(w[i,:]*x-k[i,:]))*(x)

    
    def g(x,i):
        return g_b(x, i, alpha, w, k, b)

    def dgalpha(tilde_y,i):       
        return -1*(dfalpha(tilde_y,i)/ f_prime(tilde_y,i))                   
      
    def dgw(tilde_y,i):       
        return -1*(dfw(tilde_y,i)/ f_prime(tilde_y,i))

    def dgk(tilde_y,i):       
        return -1*(dfk(tilde_y,i)/ f_prime(tilde_y,i))

    def dgb(tilde_y,i):       
        return -1*(dfb(tilde_y,i)/ f_prime(tilde_y,i))

    # FORWARD PASS

    tilde_y_tm = np.zeros((N, P+1))   #!LEq(7a) from the paper # here i_prime is used instead of i
    for i_prime in range(N):
        for p in range(1,P+1):
            assert t-p >= 0
            z_i_tmp = z_data[i_prime, t-p]
            tilde_y_tm[i_prime, p] = g(z_i_tmp,i_prime)

    # hat_y_t2 = np.zeros((N)) #!LEq(7b)
    # for i_prime in range(N): 
    #     for p in range(1,P+1):
    #         for j in range(N):
    #              hat_y_t2[i_prime] =  hat_y_t2[i_prime] + A[i_prime,j,p-1]*tilde_y_tm[j,p]
    hat_y_t =   np.zeros((N)) #!LEq(7b)
    for p in range(1,P+1):  #this equation is just to compare different looping and matrix systems. of theq 7b
        hat_y_t = hat_y_t + A[:,:,p-1]@tilde_y_tm[:,p]
    # if not np.linalg.norm(hat_y_t - hat_y_t2) < 1e-5:
    #     print("error in looping and matrix")
    #     pdb.set_trace()
    
    hat_z_t = np.zeros((N)) #!LEq(7c)
    for i_prime in range(N):    
        hat_z_t[i_prime] = f(hat_y_t[i_prime],i_prime)       
     
    # computing cost #!LEq(7d) and vector S #!LEq(8)

    S = 2*( hat_z_t - z_data[:,t])                      
    cost_u = np.sum(np.square(S[:]/2))

    
    dc_dalpha = np.zeros((N,M))
    dc_dw =  np.zeros((N,M))
    dc_dk = np.zeros((N,M))
    dc_db = np.zeros((N))
    dC_dA = np.zeros((N,N,P)) 

    if (onlyForward==False):
        # BACKWARD PASS
        # (You can use your functions f_prime, etc)
        #!Leq(17) from the paper 
        #!LEq(13) from the paper

        #precomputation of f_prime(hat_y_t[n],n):
        f_prime_hat_y_t = np.zeros((N))
        f_prime_tilde_y_tm = np.zeros((N,P+1))
        for n in range(N):
            f_prime_hat_y_t[n] = f_prime(hat_y_t[n],n)
            for p in range(P+1):
                f_prime_tilde_y_tm[n,p] = f_prime(tilde_y_tm[n,p], n)

        for i in range(N):


            for n in range(N):
                    dc_dalpha_i_a = 0
                    for p in range(1,P+1):
                        dgalpha_i_p = - dfalpha(tilde_y_tm[i,p],i)/ f_prime_tilde_y_tm[i,p]
                                #dc_dalpha_i_a = dc_dalpha_i_a   + A[n,i,p-1]*dgalpha(tilde_y_tm[i,p],i) 
                        dc_dalpha_i_a = dc_dalpha_i_a + A[n,i,p-1]*dgalpha_i_p                
                    dc_dalpha[i,:] = dc_dalpha[i,:]  + S[n]*f_prime_hat_y_t[n]*dc_dalpha_i_a     
            
            dc_dalpha[i,:] = dc_dalpha[i,:]  + S[i]*dfalpha(hat_y_t[i],i)
            
            for n in range(N):
                dc_dw_i_a = 0
                for p in range(1,P+1): 
                    dgw_i_p = - dfw(tilde_y_tm[i,p],i)/ f_prime_tilde_y_tm[i,p]
                    #dc_dw_i_a = dc_dw_i_a + A[n,i,p-1]*dgw(tilde_y_tm[i,p],i)
                    dc_dw_i_a = dc_dw_i_a + A[n,i,p-1]*dgw_i_p
                dc_dw[i,:] = dc_dw[i,:] + S[n]*f_prime_hat_y_t[n]*dc_dw_i_a 
            dc_dw[i,:] = dc_dw[i,:] + S[i]*dfw(hat_y_t[i],i)

            for n in range(N):
                dc_dk_i_a = 0
                for p in range(1,P+1): 
                    dgk_i_p = - dfk(tilde_y_tm[i,p],i)/ f_prime_tilde_y_tm[i,p]
                    #dc_dk_i_a = dc_dk_i_a + A[n,i,p-1]*dgk(tilde_y_tm[i,p],i)
                    dc_dk_i_a = dc_dk_i_a + A[n,i,p-1]*dgk_i_p
                dc_dk[i,:]= dc_dk[i,:]+ S[n]*f_prime_hat_y_t[n]*dc_dk_i_a 
            dc_dk[i,:] = dc_dk[i,:]+ S[i]*dfk(hat_y_t[i],i)

            # for n in range(N):
            #     dc_db_i_a = 0
            #     for p in range(1,P+1): 
            #         dc_dbi_a = dc_db_i_a + A[n][i][p-1]*dgb(z_data[i][t-p],i)
            #     dc_db[i] = dc_db[i] + S[n]*f_prime_hat_y_t[n]*dc_dbi_a 
            # dc_db[i]= dc_db[i] +  S[i]*dfb(hat_y_t[i],i)
        

        dC_dA = np.zeros((N,N,P))
        for i in range(N):
            for j in range(N):
                for p in range(1,P+1): 
                    dC_dA[i,j,p-1] = S[i]*f_prime_hat_y_t[i]*tilde_y_tm[j, p] 

        if np.isnan(dc_dalpha).any() or np.isinf(dc_dalpha).any():
            print('ERR: found inf or nan in dc_dalpha')
            pdb.set_trace()


    return dC_dA, dc_dalpha, dc_dw, dc_dk, dc_db, cost_u



