import numpy as np
import pdb

from compute_gradients_quick import compute_gradients 
#from compute_gradients_jacobian import compute_gradients as cgj
from projection_simplex import projection_simplex_sort as proj_simplex

def update_var_coefficients_old(eta, dC_dA, A_l,lamda): 
    # Q: why named as A_l ?
    # A: because I copied this code from the update_params_linear.
    # The input variable has been renamed as A_in
    # in update_var_coefficients (see below)

    # TODO: refactor this function using vectorization 
    # (avoiding for loops, which are slow)
    # The TODO is done in the new update_var_coefficients (see below)
    N,N,P = dC_dA.shape
    for i1 in range(N):
        for i2 in range(N):
            for p in range(P):
                fw = A_l[i1,i2,p] - eta*dC_dA[i1,i2,p]
                if fw == 0:
                    A_l[i1,i2,p] = 0
                else:
                    a1 = 1 - eta*lamda/(abs(fw))
                    A_l[i1,i2,p]  = fw * max(0,a1)
    return A_l

def update_var_coefficients(eta, dC_dA, A_in,lamda):
    fw = A_in - eta*dC_dA
    a1 = np.zeros(fw.shape)
    a1[fw != 0] = 1 - eta*lamda/(abs(fw[fw != 0]))
    
    return fw * np.maximum(0, a1)

def project_w(w, min_w = 0.001):
    w[w < min_w] = min_w
    return w

def project_alpha(alpha, z_range):
    N, M = alpha.shape
    if np.isnan(alpha).any() or np.isinf(alpha).any():
        print('ERR: found inf or nan in alpha')
        pdb.set_trace()
    
    for i in range(N):
        if (alpha[i,:].sum()  !=  z_range[i]) or (alpha[i,:]<0).any():            
            try: #projection using the code found online
                alpha[i][:] = proj_simplex(alpha[i][:], z_range[i])
            except Exception as exc:
                print('ERR: exception at proj_simplex')
                pdb.set_trace()
            if abs(np.sum(alpha[i][:])-z_range[i]) > 1e-5:
                print('ERR: projection failed!')
                pdb.set_trace()

        if (alpha[i,:]<-1e-8).any():
            print('ERR:some alphas are negative (with significant abs value)')
            pdb.set_trace()

        if np.isnan(alpha[i,:]).any() or np.isinf(alpha[i,:]).any():
            print('ERR: found inf or nan in alpha')
            pdb.set_trace()
            
    # At this point, every entry of alpha is either nonnegative or has very 
    # small absolute value. This line will make those negative entries positive 
    alpha[alpha<0] = 0
    
    return alpha

def update_params(eta, z_data, A, alpha, w, k, b, gamma, t, z_range,lamda,dict_v):
    #Luismi: This function has not been documented yet.,
    N,N2,P = A.shape
    N3, M  = k.shape
    assert N3 == N2 == N, "incompatible dimensionality"
    
    dc_dA, dc_dalpha, dc_dw, dc_dk, dc_db, sqerr = \
        compute_gradients( z_data, A, alpha, w, k, b, gamma,t)
    # dC_dA2, dc_dalpha2, dc_dw2, dc_dk2, dc_db2, sqerr2 = \
    #     cgj( z_data, A, alpha, w, k, b, t)
    # pdb.set_trace()
    if not(np.isfinite(dc_dalpha).all):
        print('some dc_dalphas are nan or inf')
        pdb.set_trace()
    
    # projected SGD (stochastic gradient descent (OPTIMIZER)
    beta = 0.9
    
    # momentum implemetation 
    
    dict_v["v_dalpha"] = beta*dict_v["v_dalpha"] + (1-beta)*dc_dalpha
    dict_v["v_dw"] =  beta*dict_v["v_dw"] + (1-beta)*dc_dw
    dict_v["v_dk"] =  beta*dict_v["v_dk"] + (1-beta)*dc_dk

    alpha = alpha - eta* dict_v["v_dalpha"]
    w     = w     - eta* dict_v["v_dw"]
    k     = k     - eta* dict_v["v_dk"]

    # (no gradient step on b because b is set as z_min)

    # Proximal gradient step:
    A = update_var_coefficients(eta, dc_dA, A,lamda)
    

    # projection of w into strictly positive values
    w = project_w(w)

    # projection of alpha
    alpha = project_alpha(alpha, z_range)
    #pdb.set_trace()

    return A, alpha, w, k, b, sqerr

