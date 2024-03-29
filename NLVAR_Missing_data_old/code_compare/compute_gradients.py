import numpy as np
import sys
sys.path.append('code_compare')
from  g_bisection import g_bisection as g_b
from f import f_param, f_prime_param, sigmoid
import pdb

def compute_gradients(z_data, A, alpha, w, k, b, t,z_true):

   
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
    
# remeber to use stable sigmoid and note the time
    
    # def sigmoid(x):
    #     sig = 1 / (1 + np.exp(-x))
    #     return sig
    # def sigmoid(x): # stable sigmoid
    #     return np.where(x >= 0, 
    #         1 / (1 + np.exp(-x)), 
    #         np.exp(x) / (1 + np.exp(x)))
            
    # def sigmoid(l):
          
    #      return  1/(1+np.exp(-l)) 

    def f_prime(x,i):
        return f_prime_param(x,i, alpha, w, k, b)

    def dgalpha(x,i):
        
        return -1*(dfalpha(x,i)/ f_prime(x,i))                   
      
    def dgw(x,i):
        
        return -1*(dfw(x,i)/ f_prime(x,i))

    def dgk(x,i):
        
        return -1*(dfk(x,i)/ f_prime(x,i))

    def dgb(x,i):
        
        return -1*(dfb(x,i)/ f_prime(x,i))
        
    
    # definitiion of f_prime  dfalha, dfw, dfk and dfb (being written as optional just to verify)
    # def f_prime(x,i): 

    #     a=0

    #     for m in range(M):
    #         a = a + alpha[i,m] * sigmoid(w[i,m]*x-k[i,m]) * (1-sigmoid(w[i,m]*x-k[i,m]))*(w[i,m])
             
    #     return a

    # def f_prime(x,i):
    #     a = 0
    #     a = alpha[i,:] * sigmoid(w[i,:]*x-k[i,:]) * (1-sigmoid(w[i,:]*x-k[i,:]))*(w[i,:])
    #     return a.sum()
    

    def dfalpha(x,i): 

        return sigmoid(w[i,:]*x-k[i,:]) 

        
    
    def dfb(x,i): 

        return 1
    
    def dfk(x,i): 

        return alpha[i,:] * sigmoid(w[i,:]*x-k[i,:]) * (1-sigmoid(w[i,:]*x-k[i,:]))*(-1)



    def dfw(x,i): 

        
        return alpha[i,:] * sigmoid(w[i,:]*x-k[i,:]) * (1-sigmoid(w[i,:]*x-k[i,:]))*(x)

    
    # core functions

    # def f(x,i): 
    #     a = b[i]
    #     a2 = 0
    #     for m in range(M):
    #         a2 = a2 + alpha[i,m] * sigmoid(w[i,m]*x-k[i,m])
    #     a2 = a2+a
    #     return a2

    def f(x,i):
        return f_param(x, i, alpha, w, k, b)

    # def g(x,i):
    #     try:
    #         return g_b(x, i, alpha, w, k, b)
    #     except:
    #         pdb.set_trace()
    #         g_b(x, i, alpha, w, k, b)
    def g(x,i):
        return g_b(x, i, alpha, w, k, b)


#not the small t it iterates over all T 


    #T80p = int(0.8*T)

   

    tilde_y_tm = np.zeros((N, P+1))   #!LEq(7a) from the paper # here i_prime is used instead of i
    for i_prime in range(N):
        for p in range(1,P+1):
            assert t-p >= 0
            z_i_tmp = z_data[i_prime, t-p]
            tilde_y_tm[i_prime, p] = g(z_i_tmp,i_prime)

    z_i_tmp_i = z_data[:,t-P:t]

    #pdb.set_trace()

    hat_y_t = np.zeros((N)) #!LEq(7b)
    for i_prime in range(N): 
        for p in range(1,P+1):
            for j in range(N):
                 hat_y_t[i_prime] =  hat_y_t[i_prime] + A[i_prime,j,p-1]*tilde_y_tm[j,p]


    hat_y_t2 =   np.zeros((N)) #!LEq(7b)
    for p in range(1,P+1):  #this equation is just to comprae different looping and matrix systems. of theq 7b
        hat_y_t2 = hat_y_t2 + A[:,:,p-1]@tilde_y_tm[:,p]
    if not np.linalg.norm(hat_y_t - hat_y_t2) < 1e-5:
        print("error in looping and matrix")
        pdb.set_trace()
    

    hat_z_t = np.zeros((N,T)) #!LEq(7c) #non need of entire matrix as we are passing only one column
    for i_prime in range(N):    
        hat_z_t[i_prime,t] = f(hat_y_t[i_prime],i_prime)       
    
   
    # computing cost #!LEq(7d) and vector S #!LEq(8)
    cost_i = [0]*T   # not use of declaring it as array as we are returning a single element
    cost_i_test = [0]*T   # not use of declaring it as array as we are returning a single element
    cost_i_val  = [0]*T

    if (t <= int(T*0.7)):
       
        cost_u = 0       

        S = 2*( hat_z_t[:,t] - z_data[:,t])  #make the modification here for the train and test error                      
        
        # for i_prime in range(N):                         #do signal reconstruciton if possible for sensor one
        #     cost_u = cost_u +  np.square(S[i_prime]/2)   
        cost_u = np.sum(np.square(S[:]/2))
          
            
        cost_i[t] = cost_u

    elif( int(T*0.7) < t <= int(T*0.8)):

        cost_u = 0
        

        S = 2*( hat_z_t[:,t] - z_data[:,t])  #make the modification here for the train and test error                      
        
        # for i_prime in range(N):                         #do signal reconstruciton if possible for sensor one
        #     cost_u = cost_u +  np.square(S[i_prime]/2)   
        cost_u = np.sum(np.square(S[:]/2))
            
        cost_i_val[t] = cost_u

    else:


        cost_u = 0
        

        S = 2*( hat_z_t[:,t] - z_data[:,t])  #make the modification here for the train and test error                      
        
        # for i_prime in range(N):                         #do signal reconstruciton if possible for sensor one
        #     cost_u = cost_u +  np.square(S[i_prime]/2)   
        cost_u = np.sum(np.square(S[:]/2))
                
        cost_i_test[t] = cost_u
        
   
   

     #for loop insdie this 
    #copy paste the prediction commands

   # Backward pass (backpropagation)
   # (You can use your functions f_prime, etc)
   #!Leq(17) from the paper 
   #!LEq(13) from the paper 
    
    
    
        

    dc_dalpha = np.zeros((N,M))
    dc_dw =  np.zeros((N,M))
    dc_dk = np.zeros((N,M))
    dc_db = np.zeros((N))
    dC_dA = np.zeros((N,N,P))    
    
    if (t < int(T*0.7)):

        for i in range(N):

        # look at the equations from the paper and change undercore i to general form carefully
            
            #pdb.set_trace()

            for n in range(N):
                dc_dalpha_i_a = 0
                for p in range(1,P+1): 
                    dc_dalpha_i_a = dc_dalpha_i_a + A[n,i,p-1]*dgalpha(tilde_y_tm[i,p],i)               
                dc_dalpha[i,:] = dc_dalpha[i,:]  + S[n]*f_prime(hat_y_t[n],n)*dc_dalpha_i_a     
            dc_dalpha[i,:] = dc_dalpha[i,:]  + S[i]*dfalpha(hat_y_t[i],i)  
            #dc_dalpha[i,:] = -1*dc_dalpha[i,:]

            
            
            for n in range(N):
                dc_dw_i_a = 0
                for p in range(1,P+1): 
                    dc_dw_i_a = dc_dw_i_a + A[n,i,p-1]*dgw(tilde_y_tm[i,p],i)
                dc_dw[i,:] = dc_dw[i,:] + S[n]*f_prime(hat_y_t[n],n)*dc_dw_i_a 
            dc_dw[i,:] = dc_dw[i,:] + S[i]*dfw(hat_y_t[i],i)

            
            
            for n in range(N):
                dc_dk_i_a = 0
                for p in range(1,P+1): 
                    dc_dk_i_a = dc_dk_i_a + A[n,i,p-1]*dgk(tilde_y_tm[i,p],i)
                dc_dk[i,:]= dc_dk[i,:]+ S[n]*f_prime(hat_y_t[n],n)*dc_dk_i_a 
            dc_dk[i,:] = dc_dk[i,:]+ S[i]*dfk(hat_y_t[i],i)

            

            # for n in range(N):
            #     dc_db_i_a = 0
            #     for p in range(1,P+1): 
            #         dc_dbi_a = dc_db_i_a + A[n][i][p-1]*dgb(z_data[i][t-p],i)
            #     dc_db[i] = dc_db[i] + S[n]*f_prime(hat_y_t[n],n)*dc_dbi_a 
            # dc_db[i]= dc_db[i] +  S[i]*dfb(hat_y_t[i],i)
        

            

        dC_dA = np.zeros((N,N,P))
        #notice that the interation over i was not there before in the previous codes 
        # check the column (:) I have doubt on that
        for i in range(N):
            for j in range(N):
                for p in range(1,P+1): 
                    dC_dA[i,j,p-1] = S[i]*f_prime(hat_y_t[n],i)*tilde_y_tm[j, p] 



        if np.isnan(dc_dalpha).any() or np.isinf(dc_dalpha).any():
            print('ERR: found inf or nan in alpha')
            pdb.set_trace()

    
    #pdb.set_trace()
    
    
    return dC_dA, dc_dalpha, dc_dw, dc_dk, dc_db, cost_i[t],cost_i_test[t],cost_i_val[t]




def compute_gradients_z(z, A, alpha, w, k, b, t, m_data, z_tilde_data, hyperparam_nu,z_true):
    
    N,N,P = A.shape
    N,T = z_tilde_data.shape

    def sigmoid(x):

        return np.where(x >= 0, 
                1 / (1 + np.exp(-x)), 
                np.exp(x) / (1 + np.exp(x)))


    # def f(x,i): 
    #     a3 = 0
    #     a3 = alpha[i,:]* sigmoid(w[i,:]*x-k[i,:])
    #     return (a3.sum() +b[i])

    def f(x,i):
        return f_param(x, i, alpha, w, k, b)
    

    # def f_prime(x,i):
    #     a = 0
    #     a = alpha[i,:] * sigmoid(w[i,:]*x-k[i,:]) * (1-sigmoid(w[i,:]*x-k[i,:]))*(w[i,:])
    #     return a.sum()

    def f_prime(x,i):
        return f_prime_param(x,i, alpha, w, k, b)

    def g(x,i):
        return g_b(x, i, alpha, w, k, b)

    def dgz(z,i):

        return 1/ (f_prime(g(z,i),i))

    

    # z_hat in INTAP paper is same as z_cap of missing data so copying the same steps below

    tilde_y_tm = np.zeros((N, P+1))   #!LEq(7a) from the paper # here i_prime is used instead of i
    
    for i_prime in range(N):
        for p in range(1,P+1):
            assert t-p >= 0
            z_i_tmp = z[i_prime, t-p]
            tilde_y_tm[i_prime, p] = g(z_i_tmp,i_prime)



    # hat_y_t = np.zeros((N)) #!LEq(7b)
    # for i_prime in range(N): 
    #     for p in range(1,P+1):
    #         for j in range(N):
    #              hat_y_t[i_prime] =  hat_y_t[i_prime] + A[i_prime,j,p-1]*tilde_y_tm[j,p]


    check_y_t =   np.zeros((N)) #!LEq(7b)
    for p in range(1,P+1):  #this equation is just to comprae different looping and matrix systems. of theq 7b
        check_y_t = check_y_t + A[:,:,p-1]@tilde_y_tm[:,p]
    # if not np.linalg.norm(hat_y_t - hat_y_t2) < 1e-5:
    #     print("error in looping and matrix")
    #     pdb.set_trace()
    

    check_z_t = np.zeros((N,T)) #!LEq(7c) #non need of entire matrix as we are passing only one column
    for i_prime in range(N):    
        check_z_t[i_prime,t] = f(check_y_t[i_prime],i_prime)  

    # here hat_z_t is same as z_cap corresponding to equation 5 on missing data
    
    
    dCt_dZ  =  np.zeros((N,T))
    dDt_dZ = np.zeros((N,T))
    dTC_dZ  =  np.zeros((N,T))

    Mt = np.count_nonzero(m_data[:,t])
    
    S = (z[:,t] -  check_z_t[:,t])
    
    assert P <= t <= T


    if (t < int(T*0.7)):

        for i in range(N):        
            dDt_dZ[i,t] = m_data[i,t]*hyperparam_nu/Mt *(z[i,t]-z_tilde_data[i,t])

        for i in range(N):
            for tau in range(t-P,t):
                if tau == t:
                    dCt_dZ[i,tau] = S[i]
                elif t - P <= tau <= t-1:
                    #dgz_v1 = 1/ f_prime(g(z[i,tau],i),i)
                    dgz_v2 = 1/ f_prime(tilde_y_tm[i, t-tau],i)
                    #print(dgz_v1 - dgz_v2) #should be zero to make sure dgz shortcut is correct
                    #pdb.set_trace()
                    dC_dZ_a = 0        
                    for n in range(N):   # do the sum calculation from here                 
                        dC_dZ_a  = dC_dZ_a - S[n]*f_prime(check_y_t[n],n)*A[n,i,t-tau-1]*dgz_v2 
                    dCt_dZ[i,tau] = dC_dZ_a

        dTC_dZ = dCt_dZ + dDt_dZ




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