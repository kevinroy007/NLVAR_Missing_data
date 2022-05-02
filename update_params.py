import numpy as np
import pdb
from compute_gradients import compute_gradients as compute_gradients_n
from compute_gradients_Compare import compute_gradients_compare
from projection_simplex import projection_simplex_sort as proj_simplex



    
def update_params(eta, z_data, A, alpha, w, k, b, t, z_range,lamda,m_p,z_data_mask):
    N,N,P = A.shape
    N,M = k.shape
    #for i in range(N):  # this way of formulation is wrong (loop should be inside backward)

    b_comparing = False # for debugging purposes. For normal execution must be false.
    if b_comparing:
        dC_dA, dc_dalpha, dc_dw, dc_dk, dc_db, cost,cost_test,hat_z_t = compute_gradients_compare( z_data, A, alpha, w, k, b, t)
    else:
        dC_dA, dc_dalpha, dc_dw, dc_dk, dc_db, cost,cost_test,cost_val = compute_gradients_n( z_data, A, alpha, w, k, b, t)
    
    # projected SGD (stochastic gradient descent (OPTIMIZER)

    if not(np.isfinite(dc_dalpha).all):
        print('some dc_dalphas are nan or inf')
        pdb.set_trace()
    alpha = alpha - eta* dc_dalpha 
    w = w - eta* dc_dw
    k = k - eta* dc_dk
    a1 = 0
    for i1 in range(N):
        for i2 in range(N):
            for p in range(P):
                fw = A[i1,i2,p] - eta*dC_dA[i1,i2,p]
                if fw == 0:
                    A[i1,i2,p] = 0
                else:
                    a1 = 1 - eta*lamda/(abs(fw))
                A[i1,i2,p]  = fw * max(0,a1)
    min_w = 0.001
    w[w < min_w] = min_w
    #w = max(0,w)
    #A1 = (1 - eta*lamda/(abs(A - eta*dC_dA)))
    #A  = (A - eta*dC_dA)*A1
    
    #pdb.set_trace()

    # b[i]    = b[i] - eta * dc_db_i TODO
    if np.isnan(alpha).any() or np.isinf(alpha).any():
        print('ERR: found inf or nan in alpha')
        pdb.set_trace()


    #PROJECTION
    
    for i in range(N):
        if (alpha[i,:].sum()  !=  z_range[i]): 
            #projection using the code found online
            try:
                alpha[i][:] = proj_simplex(alpha[i][:], z_range[i])
            except Exception as exc:
                print('ERR: exception at proj_simplex')
                pdb.set_trace()
            if abs(np.sum(alpha[i][:])-z_range[i]) > 1e-5:
                print('ERR: projection failed!'); pdb.set_trace()

        if (alpha[i,:]<-1e-8).any():
            print('ERR:some alphas are negative (with significant abs value)')
            pdb.set_trace()

        if np.isnan(alpha[i,:]).any() or np.isinf(alpha[i,:]).any():
            print('ERR: found inf or nan in alpha')
            pdb.set_trace()
            
    alpha[alpha<0] = 0

            #kevins projection will not be used. We can keep the code here for comparison       
            # alpha1 = cp.Variable(M)     
            # cost_i2 = cp.sum_squares(alpha1 - alpha[i,:])
            # obj = cp.Minimize(cost_i2)

            # constr = [sum(alpha1) == z_range[i]]         
            # opt_val = cp.Problem(obj,constr).solve()    
            # alpha_cvxpy =  np.transpose(alpha1.value)

            #if (np.abs(alpha[i][:]-alpha_cvxpy)>1e-5).any():
            #   print('ERR: projections do not coincide!!'); pdb.set_trace()

    
    #pdb.set_trace()
    

    return A, alpha, w, k, b, cost,cost_test,cost_val
