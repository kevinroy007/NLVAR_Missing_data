import numpy as np
import pdb
from compute_gradients import compute_gradients_z 


def update_z_missing(eta, z, A, alpha, w, k, b, t, m_data, z_tilde_data):
    N,N,P = A.shape
    N,M = k.shape
    #for i in range(N):  # this way of formulation is wrong (loop should be inside backward)

    
    z,cost_missing,dTC_dZ = compute_gradients_z( z, A, alpha, w, k, b, t, m_data, z_tilde_data)
    
#for i in range(N):

    for tau in range(t-P,t):
        z[:,tau] = z[:,tau] - eta* dTC_dZ[:,tau]

    # projected SGD (stochastic gradient descent (OPTIMIZER)
    if np.isnan(z).any() or np.isinf(z).any():
        print('ERR: found inf or nan in alpha')
        pdb.set_trace()
    

    return z,cost_missing
