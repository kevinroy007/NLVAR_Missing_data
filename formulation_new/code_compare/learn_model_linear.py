import numpy as np
import pdb
from compute_gradients_Linear import compute_gradients as compute_gradients_l
from compute_gradients_quick import compute_gradients as compute_gradients_nl
from update_params import update_var_coefficients
# need to modify yet making 
#data generation remain Nonlinear
#investigate the need for updationg parameters for matrix A
#try to work it out for on your work sheet on ipad

def learn_model_DEPRECATED(NE,z_data, A_l,eta,lamda): #TODO: make A, alpha, w, k, b optional
    
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

        zz = np.zeros((N, 1))
        zb = np.zeros(N)
        for t in range(P, T):   
            if (t <= int(T*0.7)): 
                # dC_dA,cost[t] = compute_gradients_l(z_data, A_l, t)   
                dC_dA, _, _, _, _, cost[t] = compute_gradients_nl(
                    z_data, A_l, zz, zz, zz, zb, t, model = 'linear')
                A_l = update_var_coefficients(eta,dC_dA,A_l,lamda)

            elif( int(T*0.7) < t <= int(T*0.8)): 
                dC_dA, _, _, _, _, cost_val[t] = compute_gradients_nl(
                    z_data, A_l, zz, zz, zz, zb, t, model = 'linear', onlyForward=True)

            else:
                dC_dA, _, _, _, _, cost_val[t] = compute_gradients_nl(
                    z_data, A_l, zz, zz, zz, zb, t, model = 'linear', onlyForward=True)

            
        v_denominators = np.sum(np.square(z_data), axis=0)

        cost_history[epoch] = sum(cost)/sum(v_denominators[0:int(0.7*T)])
        cost_history_test[epoch] = sum(cost_test)/sum(v_denominators[int(0.7*T):int(0.9*T)])
        cost_history_val[epoch] = sum(cost_val)/sum(v_denominators[int(0.9*T):-1])

        print("training NMSE =", cost_history[epoch])


    return cost_history,cost_history_test,A_l,cost_history_val