import numpy as np
import torch, pdb
torch.set_num_threads(1) 
from update_params import *
from update_param_genie import update_params_genie
from compute_gradients_quick import compute_gradients
from basic_functions import g_tensor, f_param_tensor
import pickle

def learn_model_genie(NE,eta,z_data,lamda,P, M,N_init,dict,alpha,w,k,b,A, NE_linearinit = 0, \
    b_normalize_after_epoch = False):
    N,T = z_data.shape
    print ("true model function initialisation")

    
    v_dalpha = np.zeros((N,M))
    v_dw = np.zeros((N,M))
    v_dk = np.zeros((N,M))

    dict_v = {"v_dalpha":v_dalpha,"v_dw":v_dw,"v_dk":v_dk}
    pdb.set_trace()

    eta = 0
    NE = 1
    lamda = 0
    NE_linearinit = 0
    b_normalize_after_epoch = False

    cost,cost_test,A_n,cost_val = \
        learn_model(NE, eta ,z_data, A, alpha, w, k, b,lamda,dict_v, model = "genie",  \
            NE_linearInit=NE_linearinit, \
            b_normalize_after_epoch = b_normalize_after_epoch) 
    
    return cost,cost_test,A_n,cost_val



def learn_model_balta(NE,eta,z_data,lamda,P, M,N_init,dict, NE_linearinit = 30, \
    b_normalize_after_epoch = False):
    N,T = z_data.shape
    print ("linear function initialisation")

    alpha = dict["alpha"]
    w = dict["w"]
    k = dict["k"]
    b = dict["b"]
    A = np.random.randn(N,N,P)
    pdb.set_trace()

    v_dalpha = np.zeros((N,M))
    v_dw = np.zeros((N,M))
    v_dk = np.zeros((N,M))

    dict_v = {"v_dalpha":v_dalpha,"v_dw":v_dw,"v_dk":v_dk}

    cost,cost_test,A_n,cost_val = \
        learn_model(NE, eta ,z_data, A, alpha, w, k, b,lamda,dict_v, \
            NE_linearInit=NE_linearinit, \
            b_normalize_after_epoch = b_normalize_after_epoch) 
    
    return cost,cost_test,A_n,cost_val

def learn_model_init(NE,eta,z_data,lamda,P, M,N_init, NE_linearinit = 30, \
    # it is observed a nonstop run for particular lambdas while running learn_model_init
    b_normalize_after_epoch = True):
    
    cost  = [0]*N_init
    cost_test = [0]*N_init
    cost_val = [0]*N_init
    cost_f = [0]*N_init
    A_n = [0]*N_init

    N,T = z_data.shape

    for i in range(N_init):

        print ("random initialisation",i)
        alpha = np.random.rand(N,M)
        w = np.random.rand(N,M) 
        k = np.random.randn(N,M)
        b = np.random.rand(N)
        A = np.random.randn(N,N,P)
       
        v_dalpha = np.zeros((N,M))
        v_dw = np.zeros((N,M))
        v_dk = np.zeros((N,M))

        dict_v = {"v_dalpha":v_dalpha,"v_dw":v_dw,"v_dk":v_dk}

        cost[i],cost_test[i],A_n[i],cost_val[i] = \
            learn_model(NE, eta ,z_data, A, alpha, w, k, b,lamda,dict_v, \
                NE_linearInit=NE_linearinit, \
                b_normalize_after_epoch = b_normalize_after_epoch) 
        
        cost_f[i] = cost[i][NE-1]
    arg_min = np.argmin(cost_f)
    
    return cost[arg_min],cost_test[arg_min],A_n[arg_min],cost_val[arg_min]

def learn_model_linear(NE,z_data, A_l,eta,lamda,dict_v):
    alpha, w, k, b = (None,)*4
    return learn_model(NE, eta, z_data, A_l, 
        alpha, w, k, b,lamda,dict_v ,model = 'linear')


def learn_model(NE, eta ,z_data, A_in, alpha, w, k, b,lamda,dict_v,
                model = 'nonlinear',  NE_linearInit = 30,
                b_normalize_after_epoch = True):
    
    N, T = z_data.shape
    N2,N3,P = A_in.shape
    assert N==N2 and N==N3
    # document inputs

    if model == 'nonlinear':

        z_maximum = np.max(z_data, 1)  
        z_minimum = np.min(z_data, 1)     
        z_range = z_maximum-z_minimum

        z_upper = z_maximum + 0.01*z_range
        b = z_minimum - 0.01*z_range
            
        z_difference = z_upper - b

        alpha = project_alpha(alpha, z_difference)
        w = project_w(w)  
        w_normalizing, k_normalizing, y_data_normalized = \
            indirectly_normalize_y(z_data, alpha, w, k, b)
        w = w_normalizing
        k = k_normalizing
        
        #pdb.set_trace()
       
        print('Finding optimal A for initial theta...')  
        # for nonlinear A initialisation using linear VAR
        _, _, A, _ = learn_model_linear(        
            NE_linearInit, y_data_normalized.numpy(), 
            np.zeros((N,N,P)), eta, lamda,dict_v)

        
    elif model == 'linear':
        alpha, w, k = (np.zeros((N, 1)),)*3
        b = np.zeros(N)
        A = A_in

    elif model == 'genie':  # currently genie model will not work commentinig b and alpha

        z_maximum = np.max(z_data, 1)  
        z_minimum = np.min(z_data, 1)     
        z_range = z_maximum-z_minimum

        z_upper = z_maximum + 0.01*z_range

        #b = z_minimum - 0.01*z_range  
            
        z_difference = z_upper - b

        #alpha = project_alpha(alpha, z_difference)
            
        A = A_in

        pdb.set_trace()

    else:
        raise Exception("unrecognized model option")
    
    nmse_history_train = np.zeros(NE)
    nmse_history_test  = np.zeros(NE)
    nmse_history_val   = np.zeros(NE)

    for epoch in range(NE):  
        sqerr = np.zeros(T)
        
        set_train_t = range(round(T*0.7))
        set_val_t   = range(set_train_t.stop, round(T*0.9))
        #make sure: intersection of set_train_t and set_val_t is empty 
        assert len(set(set_train_t) & set(set_val_t))==0 
        # test set is defined as every remaining sample
        set_test_t  = [t for t in range(P, T) \
            if t not in set_train_t and t not in set_val_t]

        for t in range(P, T): #training iterations
            if t in set_train_t:
                if model == 'nonlinear':   
                    A, alpha, w, k, b, _ = update_params(
                        eta, z_data, A, alpha, w, k, b, t, z_difference,lamda,dict_v)
                elif model == 'linear':
                    dC_dA, _, _, _, _, _ = compute_gradients(
                        z_data, A, alpha, w, k, b, t, model = 'linear')
                    A = update_var_coefficients(eta, dC_dA, A, lamda)

                elif model == "genie":
                    A, alpha, w, k, b, _ = update_params_genie(
                        eta, z_data, A, alpha, w, k, b, t, z_difference,lamda,dict_v)
                     
        
        if model == 'nonlinear' and b_normalize_after_epoch:    # why this ???
            w,k,_ = indirectly_normalize_y(z_data, alpha, w, k, b)
           
        for t in range(P, T): # squared error evaluation
            dC_dA, _, _, _, _, sqerr[t] = compute_gradients(
                z_data, A, alpha, w, k, b, t, model = model, onlyForward=True)             
    
        v_den = np.sum(np.square(z_data), axis=0)
        nmse_history_train[epoch] = sum(sqerr[set_train_t])/sum(v_den[set_train_t])
        nmse_history_val[epoch]   = sum(sqerr[set_val_t])  /sum(v_den[set_val_t]  )
        nmse_history_test[epoch]  = sum(sqerr[set_test_t]) /sum(v_den[set_test_t] )

        print(
            model, "epoch {:0>4d}  ".format(epoch),
            "NMSE: train {:.4f} ".format(nmse_history_train[epoch]),
            "val {:.4f} ".format(nmse_history_val[epoch]),
            "test {:.4f}".format(nmse_history_test[epoch]))
    
    dict_learned_params = {"alpha":alpha,"w":w,"k":k,"b":b}
    return  nmse_history_train,nmse_history_test,A,nmse_history_val

def indirectly_normalize_y(z_data, alpha, w, k, b):   #what is this ??
    y_data = g_tensor(
            f_param_tensor, torch.tensor(z_data), torch.tensor(alpha), 
            torch.tensor(w), torch.tensor(k), torch.tensor(b))
    sigma = np.std(y_data.numpy(), axis=1)
    mu    = np.mean(y_data.numpy(), axis=1)
    w_normalizing = sigma.reshape(-1,1)*w
    k_normalizing = k - w*mu.reshape(-1,1)
    y_data_normalized = g_tensor(
            f_param_tensor, torch.tensor(z_data), torch.tensor(alpha), 
            torch.tensor(w_normalizing), torch.tensor(k_normalizing), 
            torch.tensor(b))
    sigma_yn = np.std(y_data_normalized.numpy(), axis=1)
    mu_yn    = np.mean(y_data_normalized.numpy(), axis=1)
    assert (abs(sigma_yn)-1 < 1e-5).all() # sigma_yn is approximately ones
    assert (abs(mu_yn)< 1e-5).all()       # mu_yn    is approximately zeros
    
    return w_normalizing,k_normalizing,y_data_normalized