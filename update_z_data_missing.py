import numpy as np
import pdb
from compute_gradients import compute_gradients as compute_gradients_n
from compute_gradients_Compare import compute_gradients_compare
from compute_gradients import compute_gradients_z as compute_gradients_m



    
def update_z_data_missing(eta, z_data, A, alpha, w, k, b, t, z_range,lamda):
    N,N,P = A.shape
    N,M = k.shape
    #for i in range(N):  # this way of formulation is wrong (loop should be inside backward)

    b_comparing = False # for debugging purposes. For normal execution must be false.
    if b_comparing:
        dC_dA, dc_dalpha, dc_dw, dc_dk, dc_db, cost,cost_test,hat_z_t = compute_gradients_compare( z_data, A, alpha, w, k, b, t)
    else:
        z_data_m = compute_gradients_m( z_data, A, alpha, w, k, b, t)
    
    # projected SGD (stochastic gradient descent (OPTIMIZER)
    if np.isnan(z_data_m).any() or np.isinf(z_data_m).any():
        print('ERR: found inf or nan in alpha')
        pdb.set_trace()
    

    return z_data_m
