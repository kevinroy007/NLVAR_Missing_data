import numpy as np
from update_params import update_params
from projection_simplex import projection_simplex_sort as proj_simplex
import pdb
from update_z_missing import update_z_missing

# def learn_model_init(NE,eta,lamda,P, M,N_init,m_data,z_tilde_data):
    
#     cost  = [0]*N_init
#     cost_test = [0]*N_init
#     cost_val = [0]*N_init
#     cost_f = [0]*N_init
#     A_n = [0]*N_init
#     cost_history_missing =  [0]*N_init
#     z = [0]*N_init
#     N,T = z_tilde_data.shape

#     for i in range(N_init):
#         print ("random initialisation",i)
        
#         z = z_tilde_data
#         alpha = np.random.rand(N,M)
#         w = np.random.rand(N,M) 
#         k = np.random.randn(N,M)
#         b = np.random.rand(N)
#         A = np.random.randn(N,N,P)
       
#         cost[i],cost_test[i],A_n[i],cost_val[i],z[i],cost_history_missing[i] = learn_model(NE, eta ,z, A, alpha, w, k, b,lamda,m_data,z_tilde_data) 
#         cost_f[i] = cost[i][NE-1]
     
#     arg_min = np.argmin(cost_f)
    
#     return cost[arg_min],cost_test[arg_min],A_n[arg_min],cost_val[arg_min],z[arg_min],cost_history_missing[arg_min]
    
    
def learn_model_init(NE,eta,lamda,P, M,N_init,m_data,z_tilde_data,hyperparam_nu, eta_z,z_true ):
     
    N,T = z_tilde_data.shape

    z = z_tilde_data     # it should be np.random.rand(N,T)
    alpha = np.random.rand(N,M)
    w = np.random.rand(N,M) 
    k = np.random.randn(N,M)
    b = np.random.rand(N)
    A = np.random.randn(N,N,P)
    
    cost,cost_test,A_n,cost_val,z,NMSE_z_train,NMSE_z_test,NMSE_z_val = learn_model(NE, eta ,z, A, alpha, w, k, b,lamda,m_data,z_tilde_data,hyperparam_nu, eta_z,z_true ) 
   
    return cost,cost_test,A_n,cost_val,z,NMSE_z_train,NMSE_z_test,NMSE_z_val
    

    

def learn_model(NE, eta ,z, A, alpha, w, k, b,lamda,m_data,z_tilde_data,hyperparam_nu, eta_z,z_true): #TODO: make A, alpha, w, k, b optional
    
    N, T = z.shape
    N2,N3,P = A.shape
    assert N==N2 and N==N3
    # document inputs
    
    #TODO randomly initializing A, alpha, w, k, b if not given

    z_maximum  = np.zeros(N)
    for i in range(N):
        z_maximum[i] = np.max(z_tilde_data[i,:])
    
    z_minimum  = np.zeros(N)
    for i in range(N):
        z_minimum[i] = np.min(z_tilde_data[i,:])
    
    z_range = z_maximum-z_minimum

    z_upper = z_maximum + 0.01*z_range
    z_lower = z_minimum - 0.01*z_range
        
    z_difference = z_upper - z_lower
    b = z_lower
    
    for i in range(N):
        alpha[i,:] = proj_simplex(alpha[i,:], z_difference[i])
        # newobject.nnl[i].alpha = alpha[i,:]
        # newobject.nnl[i].b = b[i]
    w[w<0] = 0    
    cost_history = np.zeros(NE)
    cost_history_test = np.zeros(NE)
    cost_history_val  = np.zeros(NE)

    NMSE_z_train = np.zeros(NE)
    NMSE_z_test = np.zeros(NE)
    NMSE_z_val = np.zeros(NE)

    for epoch in range(NE):  
        print("Non linear epoch",epoch)
        cost = np.zeros(T)
        cost_test = np.zeros(T)
        cost_val = np.zeros(T)

        MSE_z_train =  np.zeros(T)
        MSE_z_test   = np.zeros(T)
        MSE_z_val =    np.zeros(T)
        #compare_f = np.zeros(T)

        for t in range(P, T):    # bi-level optimisation problem first solves for paramteres and then z,
            #here you are not comparing with any other model 

            #pdb.set_trace() 
            # hat_z_t = np.zeros(N)    
            # v_z_hat = np.zeros(N) 
            A, alpha, w, k, b, cost[t],cost_test[t],cost_val[t] = update_params(eta, z, A, alpha, w, k, b, t, z_difference,lamda,z_true)
            #z_data_m = update_params(eta, z_data, A, alpha, w, k, b, t, z_difference,lamda,m_p,z_data_mask)
            # pdb.set_trace()
            # v_z_hat = newobject.forward(z_data)
        #print("initiated z updation")
        ######################################################################################

        for t in range(P,T):
            #print("time stamp for z updation",t)   # returning error between learned parameters

            z,MSE_z_train[t],MSE_z_test[t],MSE_z_val[t] = update_z_missing(eta_z, z, A, alpha, w, k, b, t, m_data, z_tilde_data,hyperparam_nu,z_true)
            
        
        #######################################################################################

        v_denominators = np.sum(np.square(z), axis=0)
        v_denominators2 = np.sum(np.square(z_true), axis=0)

        cost_history[epoch] = sum(cost)/sum(v_denominators[0:int(0.7*T)])
        cost_history_test[epoch] = sum(cost_test)/sum(v_denominators[int(0.7*T):int(0.9*T)])
        cost_history_val[epoch] = sum(cost_val)/sum(v_denominators[int(0.9*T):-1])
        
        NMSE_z_train[epoch] = sum(MSE_z_train)/sum(v_denominators2)
        NMSE_z_test[epoch] = sum(MSE_z_test)/sum(v_denominators2)
        NMSE_z_val[epoch] = sum(MSE_z_val)/sum(v_denominators2)
    
    #pdb.set_trace()
    return  cost_history,cost_history_test,A,cost_history_val,z,NMSE_z_train,NMSE_z_test,NMSE_z_val


    
 