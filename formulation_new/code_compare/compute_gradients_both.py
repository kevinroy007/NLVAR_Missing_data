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
    
    # OLD Function definitions
    def f(x,i):
        a3 = alpha[i,:]* sigmoid(w[i,:]*x-k[i,:])
        return a3.sum() +b[i]

    def f_prime(x,i):
        a = alpha[i,:] * sigmoid(w[i,:]*x-k[i,:]) * (1-sigmoid(w[i,:]*x-k[i,:]))*(w[i,:])
        return a.sum()

    def g(x,i):
        return g_b(x, i, alpha, w, k, b, tol = 1e-12)

    # NEW function definitions

    def ft(x, compute_f_prime=False):
        return f_param_tensor(
            x, torch.tensor(alpha), torch.tensor(w), torch.tensor(k), 
            torch.tensor(b), compute_f_prime)

    def gt(z):
        return g_tensor(
            f_param_tensor, z, torch.tensor(alpha), torch.tensor(w), 
            torch.tensor(k), torch.tensor(b), tol=1e-12)

    # gradient of g w.r.t theta     #!LEq(17)

    def dfalpha(x,i): 
        return sigmoid(w[i,:]*x-k[i,:]) 
   
    def dfw(x,i):         
        return alpha[i,:] * sigmoid(w[i,:]*x-k[i,:]) * (1-sigmoid(w[i,:]*x-k[i,:]))*(x)
    
    def dfk(x,i): 
        return alpha[i,:] * sigmoid(w[i,:]*x-k[i,:]) * (1-sigmoid(w[i,:]*x-k[i,:]))*(-1)

    def dfb(x,i): 
        return 1

    def gradients_f(t_x_in):
        tx_shape = list(t_x_in.shape)
        tx_shape.insert(1, 1)
        t_x = t_x_in.reshape(tx_shape)    
        param_shape = [1]*t_x.dim()
        param_shape[0:2] = w.shape
        alpha2 = torch.tensor(alpha).reshape(param_shape)
        w2 =     torch.tensor(    w).reshape(param_shape)
        k2 =     torch.tensor(    k).reshape(param_shape)
        #copied from basic_functions.f_param_tensor

        s_out = torch.sigmoid(w2 * t_x - k2)
        grad_alpha =  s_out
        grad_w     =  alpha2 * s_out * (1-s_out) * t_x
        grad_k     = -alpha2 * s_out * (1-s_out)
        return grad_alpha, grad_w, grad_k

    # FORWARD PASS

    tilde_y_tm = np.zeros((N, P+1))   #!LEq(7a) from the paper # here i_prime is used instead of i
    for i_prime in range(N):
        for p in range(1,P+1):
            assert t-p >= 0
            z_i_tmp = z_data[i_prime, t-p]
            tilde_y_tm[i_prime, p] = g(z_i_tmp,i_prime)
    tilde_y_tm2 = torch.zeros((N, P+1))
    z_in = np.flip(z_data[:,t-P:t], 1).copy()
    tilde_y_tm2[:,1:] = gt(torch.tensor(z_in))

    
    hat_y_t =   np.zeros((N)) #!LEq(7b)
    for p in range(1,P+1):  #this equation is just to compare different looping and matrix systems. of theq 7b
        hat_y_t = hat_y_t + A[:,:,p-1]@tilde_y_tm[:,p]
    hat_y_t2 = torch.zeros(N)
    for p in range(1,P+1):  #this equation is just to compare different looping and matrix systems. of theq 7b
        hat_y_t2 = hat_y_t2 + torch.tensor(A[:,:,p-1]).float()@tilde_y_tm2[:,p]
    
    
    hat_z_t = np.zeros((N)) #!LEq(7c)
    for i_prime in range(N):    
        hat_z_t[i_prime] = f(hat_y_t[i_prime],i_prime)
    if onlyForward:
        hat_z_t2 = ft(hat_y_t2)
    else:
        tilde_y_tm2[:,0] = hat_y_t2
        aux_tensor, aux_tensor_prime = ft(tilde_y_tm2, compute_f_prime=True)
        hat_z_t2 = aux_tensor[:,0]
        f_prime_hat_y_t2 = aux_tensor_prime[:,0]
        f_prime_tilde_y_tm2 = aux_tensor_prime.clone()
        f_prime_tilde_y_tm2[:,0] = 0
    
    # computing cost #!LEq(7d) and vector S #!LEq(8)

    S = 2*( hat_z_t - z_data[:,t])                      
    cost_u = np.sum(np.square(S[:]/2))

    S2 = 2*( hat_z_t2 - z_data[:,t])
    cost_u = torch.sum(torch.square(S/2))
    
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
             
              
        grad_alpha, grad_w, grad_k = gradients_f(tilde_y_tm2)   
        dc_dalpha2 = torch.zeros(N,M)
        dc_dk2     = torch.zeros(N,M)
        dc_dw2     = torch.zeros(N,M)
        for i in range(N):
            # ALPHA
            for n in range(N):
                dc_dalpha_i_a = 0
                for p in range(1,P+1):
                    dgalpha_i_p = - dfalpha(tilde_y_tm[i,p],i)/ f_prime_tilde_y_tm[i,p]
                    #dc_dalpha_i_a = dc_dalpha_i_a   + A[n,i,p-1]*dgalpha(tilde_y_tm[i,p],i) 
                    dc_dalpha_i_a = dc_dalpha_i_a + A[n,i,p-1]*dgalpha_i_p                
                dc_dalpha[i,:] = dc_dalpha[i,:]  + S[n]*f_prime_hat_y_t[n]*dc_dalpha_i_a     
            dc_dalpha[i,:] = dc_dalpha[i,:]  + S[i]*dfalpha(hat_y_t[i],i)
            
            for n in range(N):
                dc_dalpha_i_a2 = 0
                for p in range(1,P+1):
                    dgalpha_i_p2 = - grad_alpha[i,:,p]/ f_prime_tilde_y_tm2[i,p]
                    dc_dalpha_i_a2 = dc_dalpha_i_a2 + A[n,i,p-1]*dgalpha_i_p2                
                dc_dalpha2[i,:] = dc_dalpha2[i,:]  + S[n]*f_prime_hat_y_t2[n]*dc_dalpha_i_a2     
            dc_dalpha2[i,:] = dc_dalpha2[i,:]  + S[i]*grad_alpha[i,:,0]

            # W
            for n in range(N):
                dc_dw_i_a = 0
                for p in range(1,P+1): 
                    dgw_i_p = - dfw(tilde_y_tm[i,p],i)/ f_prime_tilde_y_tm[i,p]
                    #dc_dw_i_a = dc_dw_i_a + A[n,i,p-1]*dgw(tilde_y_tm[i,p],i)
                    dc_dw_i_a = dc_dw_i_a + A[n,i,p-1]*dgw_i_p
                dc_dw[i,:] = dc_dw[i,:] + S[n]*f_prime_hat_y_t[n]*dc_dw_i_a 
            dc_dw[i,:] = dc_dw[i,:] + S[i]*dfw(hat_y_t[i],i)

            for n in range(N):
                dc_dw_i_a2 = 0
                for p in range(1,P+1):
                    dgw_i_p2 = - grad_w[i,:,p]/ f_prime_tilde_y_tm2[i,p]
                    dc_dw_i_a2 = dc_dw_i_a2 + A[n,i,p-1]*dgw_i_p2                
                dc_dw2[i,:] = dc_dw2[i,:]  + S[n]*f_prime_hat_y_t2[n]*dc_dw_i_a2     
            dc_dw2[i,:] = dc_dw2[i,:]  + S[i]*grad_w[i,:,0]
            
            # K
            for n in range(N):
                dc_dk_i_a = 0
                for p in range(1,P+1): 
                    dgk_i_p = - dfk(tilde_y_tm[i,p],i)/ f_prime_tilde_y_tm[i,p]
                    #dc_dk_i_a = dc_dk_i_a + A[n,i,p-1]*dgk(tilde_y_tm[i,p],i)
                    dc_dk_i_a = dc_dk_i_a + A[n,i,p-1]*dgk_i_p
                dc_dk[i,:]= dc_dk[i,:]+ S[n]*f_prime_hat_y_t[n]*dc_dk_i_a 
            dc_dk[i,:] = dc_dk[i,:]+ S[i]*dfk(hat_y_t[i],i)

            for n in range(N):
                dc_dk_i_a2 = 0
                for p in range(1,P+1):
                    dgk_i_p2 = - grad_k[i,:,p]/ f_prime_tilde_y_tm2[i,p]
                    dc_dk_i_a2 = dc_dk_i_a2 + A[n,i,p-1]*dgk_i_p2                
                dc_dk2[i,:] = dc_dk2[i,:]  + S[n]*f_prime_hat_y_t2[n]*dc_dk_i_a2     
            dc_dk2[i,:] = dc_dk2[i,:]  + S[i]*grad_k[i,:,0]
            
            
            # for n in range(N):
            #     dc_db_i_a = 0
            #     for p in range(1,P+1): 
            #         dc_dbi_a = dc_db_i_a + A[n][i][p-1]*dgb(z_data[i][t-p],i)
            #     dc_db[i] = dc_db[i] + S[n]*f_prime_hat_y_t[n]*dc_dbi_a 
            # dc_db[i]= dc_db[i] +  S[i]*dfb(hat_y_t[i],i)
        

        dC_dA = np.zeros((N,N,P))
        dC_dA2 = torch.zeros(N,N,P)
        for i in range(N):
            for j in range(N):
                for p in range(1,P+1): 
                    dC_dA[i,j,p-1]  = S[i]*f_prime_hat_y_t[i]*tilde_y_tm[j, p]
                    dC_dA2[i,j,p-1] = S[i]*f_prime_hat_y_t2[i]*tilde_y_tm2[j, p]
        #pdb.set_trace()
        
        if np.isnan(dc_dalpha).any() or np.isinf(dc_dalpha).any():
            print('ERR: found inf or nan in dc_dalpha')
            pdb.set_trace()


    return dC_dA, dc_dalpha, dc_dw, dc_dk, dc_db, cost_u

