import numpy as np
import pdb, sys

from basic_functions import *
from g_bisection import g_bisection as g_b
import torch

# sys.path.append(sys.path[0]+'/..')
# from formulation_B.basic_functions import plot_the_function

def compute_gradients(
    z, A, alpha, w, k, b, gamma, t, 
    onlyForward = False, g_tol = 1e-6, model='nonlinear'):  

    #print(t)
# COMPUTE_GRADIENTS gets gradients for cost at time t 
# w.r.t. parameters at subnetwork i
# INPUT VARIABLES
# t: integer, is the time instant
# i: integer, is the subnetwork index
# z_data N x T array, containing the given z data
    # N: number of sensors
    # T: number of time instants
    N, T = z.shape
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
    # def f(x,i):
    #     a3 = alpha[i,:]* sigmoid(w[i,:]*x-k[i,:])
    #     return a3.sum() +b[i]

    # def f_prime(x,i):
    #     a = alpha[i,:] * sigmoid(w[i,:]*x-k[i,:]) \
    #         * (1-sigmoid(w[i,:]*x-k[i,:]))*(w[i,:])
    #     return a.sum()

    # def g(x,i):
    #     return g_b(x, i, alpha, w, k, b, tol = g_tol)

    # NEW function definitions

    def ft(x, compute_f_prime=False):
        return f_param_tensor(
            x, torch.tensor(alpha), torch.tensor(w), torch.tensor(k), 
            torch.tensor(b), gamma, compute_f_prime)

    def gt(z):
        return g_tensor(
            f_param_tensor, z, torch.tensor(alpha), torch.tensor(w), 
            torch.tensor(k), torch.tensor(b), gamma, tol=g_tol)

    def idt(x, compute_f_prime=False): #identity and derivative (ones)
        if compute_f_prime:
            return x, torch.ones(x.shape)
        else:
            return x

    # gradient of g w.r.t theta     #!LEq(17)

    # def dfalpha(x,i): 
    #     return sigmoid(w[i,:]*x-k[i,:]) 
   
    # def dfw(x,i):         
    #     return alpha[i,:] * sigmoid(w[i,:]*x-k[i,:]) \
    #       * (1-sigmoid(w[i,:]*x-k[i,:]))*(x)
    
    # def dfk(x,i): 
    #     return alpha[i,:] * sigmoid(w[i,:]*x-k[i,:]) \
    #       * (1-sigmoid(w[i,:]*x-k[i,:]))*(-1)

    # def dfb(x,i): 
    #     return 1

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

    if model == 'linear':
        ft = idt
        gt = idt

    # FORWARD PASS

    # tilde_y_tm = np.zeros((N, P+1))   #!LEq(7a) from the paper
    # for i_prime in range(N):
    #     for p in range(1,P+1):
    #         assert t-p >= 0
    #         z_i_tmp = z_data[i_prime, t-p]
    #         tilde_y_tm[i_prime, p] = g(z_i_tmp,i_prime)
    tilde_y_tm2 = torch.zeros((N, P+1))
    z_in = np.flip(z[:,t-P:t], 1).copy()   #why copy is used 
    tilde_y_tm2[:,1:] = gt(torch.tensor(z_in))

    
    # hat_y_t =   np.zeros((N)) #!LEq(7b)
    # for p in range(1,P+1):  #this equation is just to compare 
    #   # different looping and matrix systems. of theq 7b
    #     hat_y_t = hat_y_t + A[:,:,p-1]@tilde_y_tm[:,p]
    hat_y_t2 = torch.zeros(N)
    for p in range(1,P+1):  #this equation is just to compare 
        # different looping and matrix systems. of theq 7b
        hat_y_t2 = hat_y_t2 + torch.tensor(A[:,:,p-1]).float()@tilde_y_tm2[:,p]
    
    
    # hat_z_t = np.zeros((N)) #!LEq(7c)
    # for i_prime in range(N):    
    #     hat_z_t[i_prime] = f(hat_y_t[i_prime],i_prime)
    if onlyForward:
        hat_z_t2 = ft(hat_y_t2)
    else:
        hat_z_t2, f_prime_hat_y_t2 = ft(hat_y_t2, compute_f_prime=True)
        _, f_prime_tilde_y_tm2 =     ft(tilde_y_tm2, compute_f_prime=True)
        f_prime_tilde_y_tm2[:,0] = 0
        #old code:
        # tilde_y_tm2[:,0] = hat_y_t2
        # aux_tensor, aux_tensor_prime = ft(tilde_y_tm2, compute_f_prime=True)
        # hat_z_t2 = aux_tensor[:,0]
        # f_prime_hat_y_t2 = aux_tensor_prime[:,0]
        # f_prime_tilde_y_tm2 = aux_tensor_prime.clone()
        # f_prime_tilde_y_tm2[:,0] = 0
    #pdb.set_trace()
    
    # computing cost #!LEq(7d) and vector S #!LEq(8)

    # S = 2*( hat_z_t2 - z_data[:,t])                      
    # cost_u = np.sum(np.square(S[:]/2))

    S2 = 2*( hat_z_t2 - z[:,t])
    cost_u = torch.sum(torch.square(S2/2))
    
    if onlyForward:
        dc_dalpha = np.zeros((N,M))
        dc_dw =  np.zeros((N,M))
        dc_dk = np.zeros((N,M))
        dc_db = np.zeros((N))
        dC_dA = np.zeros((N,N,P)) 

    else:
        # BACKWARD PASS
        # (You can use your functions f_prime, etc)
        #!Leq(17) from the paper 
        #!LEq(13) from the paper

        #precomputation of f_prime(hat_y_t[n],n):
        # f_prime_hat_y_t = np.zeros((N))
        # f_prime_tilde_y_tm = np.zeros((N,P+1))
        # for n in range(N):
        #     f_prime_hat_y_t[n] = f_prime(hat_y_t[n],n)
        #     for p in range(P+1):
        #         f_prime_tilde_y_tm[n,p] = f_prime(tilde_y_tm[n,p], n)
             
        
        # def obtain_dCd(A, N, M, P, f_prime_hat_y_t2, 
        #                f_prime_tilde_y_tm2, S2, grad_theta):
        #     dc_dtheta2 = torch.zeros(N,M)
        #     for i in range(N):
        #         for n in range(N):
        #             dc_dtheta_i_a2 = 0
        #             for p in range(1,P+1):
        #                 dgtheta_i_p2 = - grad_theta[i,:,p] \
        #                   / f_prime_tilde_y_tm2[i,p]
        #                 dc_dtheta_i_a2 = dc_dtheta_i_a2 \
        #                   + A[n,i,p-1]*dgtheta_i_p2                
        #             dc_dtheta2[i,:] = dc_dtheta2[i,:] \
        #                 + S2[n]*f_prime_hat_y_t2[n]*dc_dtheta_i_a2     
        #         dc_dtheta2[i,:] = dc_dtheta2[i,:]  + S2[i]*grad_theta[i,:,0]
        #     return dc_dtheta2
        
        if model == 'nonlinear':
            def obtain_dCd2(
                A, N, M, P, f_prime_hat_y_t2, f_prime_tilde_y_tm2, 
                S2, grad_theta_hat_y, grad_theta_tilde_y):         
                # index order: i, m, n, p
                A_view = torch.tensor(A).permute(1, 0, 2)
                my_t = -S2.reshape(1, 1, N, 1) \
                    * f_prime_hat_y_t2.reshape(1, 1, N, 1)\
                    * A_view.reshape(N,1,N,P)* \
                    grad_theta_tilde_y[:,:,1:].reshape(N,M,1,P) \
                    /f_prime_tilde_y_tm2[:,1:].reshape(N,1,1,P)
                return S2.reshape(N, 1)*grad_theta_hat_y + my_t.sum((2,3))
            
            grad_alpha_tilde_y, grad_w_tilde_y, grad_k_tilde_y = gradients_f(tilde_y_tm2)
            grad_alpha_hat_y, grad_w_hat_y, grad_k_hat_y = gradients_f(hat_y_t2)
        
        elif model == 'linear':
            def obtain_dCd2(A, N, M, P, f_prime_hat_y_t2, f_prime_tilde_y_tm2, S2,
                             grad_theta_hat_y, grad_theta_tilde_y):
                return torch.zeros(N,M)
            grad_alpha_hat_y, grad_w_hat_y, grad_k_hat_y,\
            grad_alpha_tilde_y, grad_w_tilde_y, grad_k_tilde_y = (None,)*6

        #dc_dalpha2 = obtain_dCd( A, N, M, P, f_prime_hat_y_t2, \
        #       f_prime_tilde_y_tm2, S2, grad_alpha)
        dc_dalpha3  = obtain_dCd2(A, N, M, P, f_prime_hat_y_t2, \
            f_prime_tilde_y_tm2, S2, grad_alpha_hat_y, grad_alpha_tilde_y)  # I think this lines wont excute if the model is linear
        #dc_dk2     = obtain_dCd( A, N, M, P, f_prime_hat_y_t2, \
        #       f_prime_tilde_y_tm2, S2, grad_k)
        dc_dk3      = obtain_dCd2(A, N, M, P, f_prime_hat_y_t2, \
            f_prime_tilde_y_tm2, S2, grad_k_hat_y, grad_k_tilde_y)
        #dc_dw2     = obtain_dCd( A, N, M, P, f_prime_hat_y_t2, \
        #       f_prime_tilde_y_tm2, S2, grad_w)
        dc_dw3      = obtain_dCd2(A, N, M, P, f_prime_hat_y_t2, \
            f_prime_tilde_y_tm2, S2, grad_w_hat_y, grad_w_tilde_y)
        dc_db3      = torch.zeros(N)
            
        def obtain_dCdA2(N, P, tilde_y_tm2, f_prime_hat_y_t2, S2):
            # index order: i, j, p
            return S2.reshape(N, 1, 1) * f_prime_hat_y_t2.reshape(N, 1, 1) \
                * tilde_y_tm2[:,1:].reshape(1, N, P)
        # dC_dA = np.zeros((N,N,P))
        dC_dA2 = obtain_dCdA2(N, P, tilde_y_tm2, f_prime_hat_y_t2, S2)
        
        # if np.isnan(dc_dalpha).any() or np.isinf(dc_dalpha).any():
        #     print('ERR: found inf or nan in dc_dalpha')
        #     pdb.set_trace()

        dc_dalpha = dc_dalpha3.numpy()
        dc_dw     = dc_dw3.numpy()
        dc_dk     = dc_dk3.numpy()
        dc_db     = dc_db3.numpy()
        dC_dA     = dC_dA2.numpy()

    return dC_dA, dc_dalpha, dc_dw, dc_dk, dc_db, cost_u




def compute_gradients_z(z, A, alpha, w, k, b, gamma, t, m_data, z_tilde_data, hyperparam_nu,z_true,g_tol = 1e-6, model='nonlinear'):
    
    N,N,P = A.shape
    N,T = z_tilde_data.shape

    def ft(x, compute_f_prime=False):
        return f_param_tensor(
            x, torch.tensor(alpha), torch.tensor(w), torch.tensor(k), 
            torch.tensor(b), gamma, compute_f_prime)
    
    def gt(z):
        return g_tensor(
            f_param_tensor, z, torch.tensor(alpha), torch.tensor(w), 
            torch.tensor(k), torch.tensor(b), gamma, tol=g_tol)

    def idt(x, compute_f_prime=False): #identity and derivative (ones)
        if compute_f_prime:
            return x, torch.ones(x.shape)
        else:
            return x


    def f(x,i):
        return f_param(x, i, alpha, w, k, b)
    

    def f_prime(x,i):
        return f_prime_param(x,i, alpha, w, k, b)

    def g(x,i):
        return g_b(x, i, alpha, w, k, b)

    def dgz(z,i):

        return 1/ (f_prime(g(z,i),i))

    

    # tilde_y_tm = np.zeros((N, P+1))   
    
    # for i_prime in range(N):
    #     for p in range(1,P+1):
    #         assert t-p >= 0
    #         z_i_tmp = z[i_prime, t-p]
    #         tilde_y_tm[i_prime, p] = g(z_i_tmp,i_prime)

    
    # check_y_t =   np.zeros((N)) 
    # for p in range(1,P+1):  
    #     check_y_t = check_y_t + A[:,:,p-1]@tilde_y_tm[:,p]
   

    # check_z_t = np.zeros((N,T)) 
    # for i_prime in range(N):    
    #     check_z_t[i_prime,t] = f(check_y_t[i_prime],i_prime)  



    ##################################################################

    tilde_y_tm2 = torch.zeros((N, P+1))
    z_in = np.flip(z[:,t-P:t], 1).copy()   
    tilde_y_tm2[:,1:] = gt(torch.tensor(z_in))

    hat_y_t2 = torch.zeros(N)
    for p in range(1,P+1):  
        hat_y_t2 = hat_y_t2 + torch.tensor(A[:,:,p-1]).float()@tilde_y_tm2[:,p]
    
    hat_z_t2, f_prime_hat_y_t2 = ft(hat_y_t2, compute_f_prime=True)
    _, f_prime_tilde_y_tm2 =     ft(tilde_y_tm2, compute_f_prime=True)
    f_prime_tilde_y_tm2[:,0] = 0
       
   
    ##################################################################
    
    
    dCt_dZ  =  np.zeros((N,T))
    dCt_dZ_v2 = np.zeros((N,T))
    dDt_dZ = np.zeros((N,T))
    dTC_dZ  =  np.zeros((N,T))

    Mt = np.count_nonzero(m_data[:,t])

    

    
    
    S = (z[:,t] - hat_z_t2.numpy())
    
    assert P <= t <= T


    if (t < int(T*0.7)):

        for i in range(N):        
            dDt_dZ[i,t] = m_data[i,t]*hyperparam_nu/Mt *(z[i,t]-z_tilde_data[i,t])

        _, f_prime_check_y_tT = ft((hat_y_t2), compute_f_prime=True)
        f_prime_check_y_t = f_prime_check_y_tT.numpy()
        for tau in range(t-P,t):
            if tau == t:               
                dCt_dZ[:,tau] = S
            elif t - P <= tau <= t-1:
                _, f_prime_tilde_y_tm = ft((tilde_y_tm2[:, t-tau]), compute_f_prime=True)
                dgz_v3 = 1 / f_prime_tilde_y_tm.numpy()
                
                for n in range(N):
                    dCt_dZ_v2[:,tau] = dCt_dZ_v2[:,tau] - S[n]*f_prime_check_y_t[n]* \
                        np.transpose(A[n,:,t-tau-1])*dgz_v3

                # for i in range(N):
                
                #     #dgz_v1 = 1/ f_prime(g(z[i,tau],i),i)
                #     dgz_v2 = 1/ f_prime(tilde_y_tm[i, t-tau],i)
                #     #print(dgz_v1 - dgz_v2) #should be zero to make sure dgz shortcut is correct
                #     #pdb.set_trace()
                #     dC_dZ_a = 0        
                #     for n in range(N):   # do the sum calculation from here                 
                #         dC_dZ_a  = dC_dZ_a - S[n]*f_prime(check_y_t[n],n)*A[n,i,t-tau-1]*dgz_v2 
                #     dCt_dZ[i,tau] = dC_dZ_a
                # pdb.set_trace()

        dTC_dZ = dCt_dZ_v2 + dDt_dZ




    # for i in range(N):
    #     for tau in range(t-P,t):
    #         if tau == t:
    #             dCt_dZ[i,tau] = S[i]
    #         elif t - P <= tau <= t:
    #             dC_dZ_a = 0        
    #             for n in range(N):   # do the sum calculation from here
    #                 dC_dZ_a  = dC_dZ_a -S[n]*f_prime(check_y_t[n],n)*A[n,i,t-tau-1]*dgz(z[i,tau],i) 
    #         dCt_dZ[i,tau] = dC_dZ_a

    cost_missing_train = 0
    cost_missing_validation = 0
    cost_missing_test = 0

    if (t < int(T*0.7)):
        
        S1 = z_true[:,t] - z[:,t]
        cost_missing_train = np.sum(np.square(S1[:]))

    elif ( int(T*0.7) < t <= int(T*0.8)):

        S1 = z_true[:,t] - z[:,t]
        cost_missing_validation = np.sum(np.square(S1[:]))

    else:
        S1 = z_true[:,t] - z[:,t]
        cost_missing_test = np.sum(np.square(S1[:]))



    return z,cost_missing_train,dTC_dZ,cost_missing_test,cost_missing_validation


