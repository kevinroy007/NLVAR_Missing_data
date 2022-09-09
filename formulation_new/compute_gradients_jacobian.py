import numpy as np
import torch
import pdb

from basic_functions import g_tensor, f_param_tensor



def compute_gradients(
    z_data, A, alpha, w, k, b, t, 
    onlyForward = False, g_tol = 1e-6, model='nonlinear'):
    def gt(z):
        return g_tensor(
            f_param_tensor, z, torch.tensor(alpha), torch.tensor(w), 
            torch.tensor(k), torch.tensor(b), tol=g_tol)
    
    def ft(x, compute_f_prime=False):
        return f_param_tensor(
            x, torch.tensor(alpha), torch.tensor(w), torch.tensor(k), 
            torch.tensor(b), compute_f_prime)

    N,N,P = A.shape
    N, M  = alpha.shape
    # FORWARD pass

    #!! old code (left for comparison, must remove)
    
    z_in = np.flip(z_data[:,t-P:t+1], 1).copy()
    tilde_y_tm = gt(torch.tensor(z_in)).numpy()  
    
    y_hat = np.zeros(N)
    dz_tm_dytilde_tm = [None,] * (P+1)
    for p in range(1, P+1):
        y_hat += A[:,:,p-1] @ tilde_y_tm[:,p]
        _, temp_fprimes = ft(torch.tensor(tilde_y_tm[:,p]), compute_f_prime=True)
        dz_tm_dytilde_tm[p] = np.diag(temp_fprimes)
    t_z_hat, t_fprimes_yhat = ft(torch.tensor(y_hat), compute_f_prime=True)
    fprimes_yhat = t_fprimes_yhat.numpy()
    z_hat = t_z_hat.numpy()
    dzhat_dyhat = np.diag(fprimes_yhat)
    C = np.sum(np.square(z_data[:,t] - z_hat))

    def df_dtheta(y):
        t_out = np.zeros((N, 3, N, M))
        t_alpha = torch.tensor(alpha).reshape([1, 1, N, M])
        t_w     = torch.tensor(w).reshape([1, 1, N, M])
        t_k     = torch.tensor(k).reshape([1, 1, N, M])
        t_y     = torch.tensor(y).reshape([N, 1, 1, 1])
        t_sigmoids = torch.sigmoid(t_w*t_y - t_k)
        t_sigprime = t_sigmoids*(1-t_sigmoids)
        for i in range(N):
            t_out[i, 0, i, :] = t_sigmoids[i, 0, i, :]
            t_out[i, 1, i, :] = t_alpha[0, 0, i, :] * y[i] * t_sigprime[i, 0, i, :]
            t_out[i, 2, i, :] = - t_alpha[0, 0, i, :] * t_sigprime[i, 0, i, :]
        
        return t_out.reshape(N, 3*N*M)

    if onlyForward:
        dC_dalpha = np.zeros((N,M))
        dC_dw =  np.zeros((N,M))
        dC_dk = np.zeros((N,M))
        dC_db = np.zeros((N))
        dC_dA = np.zeros((N,N,P)) 

    else: # BACKWARD pass
        S_T = 2*(z_hat - z_data[:,t]).transpose()
        mySum_theta = np.zeros((N, 3*N*M))
        for p in range(1, P+1):
            dz_tmp_dtheta = df_dtheta(tilde_y_tm[:,p]) 
            mySum_theta += A[:,:,p-1] @ \
                np.linalg.inv(dz_tm_dytilde_tm[p]) @ dz_tmp_dtheta
        dzhat_dtheta = df_dtheta(y_hat)
        dC_dtheta = S_T @ (dzhat_dtheta - dzhat_dyhat @ mySum_theta)
        t_dCdtheta = dC_dtheta.reshape([3, N, M])
        dC_dalpha = t_dCdtheta[0,:,:]
        dC_dw     = t_dCdtheta[1,:,:]
        dC_dk     = t_dCdtheta[2,:,:]
        dC_db     = np.zeros(N)

        # dC_dA = np.zeros((N, N, P))
        # for i in range(N):
        #     for j in range(N):
        #         for p in range(P):
        #             dC_dA[i, j, p] = 2*(z_hat[i] - z_data[i,t]) * fprimes_yhat[i] \
        #                 * tilde_y_tm[j, p+1]
        
        dC_dA2 = S_T.reshape(N, 1, 1) * fprimes_yhat.reshape(N, 1, 1) \
                * tilde_y_tm[:,1:].reshape(1, N, P)
        

    return dC_dA2, dC_dalpha, dC_dw, dC_dk, dC_db, C


