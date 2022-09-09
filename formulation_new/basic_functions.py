import numpy as np
import warnings, pdb
import torch
# defined sigmoid without stability clause because stable sigmoid is slow...

b_use_stable_sigmoid = True

if b_use_stable_sigmoid:

        def sigmoid(x): # stable sigmoid
                return np.where(x >= 0, 
                        1 / (1 + np.exp(-x)), 
                        np.exp(x) / (1 + np.exp(x)))
else:
        warnings.warn("Using unstable sigmoid...")
        def sigmoid(x):
                sig = 1 / (1 + np.exp(-x))
                return sig

def f_param(x,i, alpha, w, k, b): 
    a3 = alpha[i,:]* sigmoid(w[i,:]*x-k[i,:]) 
    return a3.sum() +b[i]#+ gamma*x

def f_param_torch(x,i, alpha, w, k, b): 
    a3 = alpha[i,:]* torch.sigmoid(w[i,:]*x-k[i,:]) 
    return a3.sum() +b[i]#+ gamma*x

def f_prime_param(x,i, alpha, w, k, b):
    a = alpha[i,:] * sigmoid(w[i,:]*x-k[i,:]) \
            * (1-sigmoid(w[i,:]*x-k[i,:]))*(w[i,:]) 
    return a.sum() #+ gamma

def f_param_tensor(t_x_in, alpha, w, k, b,gamma, produce_f_prime = False):
        #t_x_in: NxT tensor, independent variable
        #alpha:  NxM tensor, parameter
        #w:      NxM tensor, parameter
        #k:      NxM tensor, parameter
        #b:      Nx1 tensor, parameter
        tx_shape = list(t_x_in.shape)
        tx_shape.insert(1, 1)
        t_x = t_x_in.reshape(tx_shape)    
        param_shape = [1]*t_x.dim()
        param_shape[0:2] = w.shape
        alpha2 = alpha.reshape(param_shape)
        w2 =         w.reshape(param_shape)
        k2 =         k.reshape(param_shape)
        b_shape = [1]*t_x_in.dim()
        b_shape[0] = param_shape[0]
        b2 = b.reshape(b_shape)
        
        sigmoid_output = torch.sigmoid(w2*t_x-k2)
        aux = alpha2 * sigmoid_output
        if produce_f_prime:
                aux_prime = alpha2 * sigmoid_output *(1 - sigmoid_output) * w2
                return aux.sum(1) + b2 + gamma*t_x_in, aux_prime.sum(1) + gamma
                #return values: 
                # f_t_x:       NxT tensor, result of applying f to t_x
                # f_prime_t_x: NxT tensor, derivative
        else:
                return aux.sum(1) + b2 + gamma*t_x_in
                #return value: 
                # f_t_x:       NxT tensor, result of applying f to t_x

def g_tensor(f_in, t_z, alpha, w, k, b,gamma, max_niter = 1000, tol=1e-6, b_trust_f_isInvertible=False):
        def f(t_z):
                return f_in(t_z, alpha, w, k, b,gamma)

        def initialBounds(t_z, f):
                t_yl = - torch.ones(t_z.shape)
                t_yu =   torch.ones(t_z.shape)
                b_l = f(t_yl) > t_z
                while b_l.any():
                        t_yl[b_l] = t_yl[b_l] * 8
                        b_l = f(t_yl) > t_z
                b_u = f(t_yu) < t_z
                while b_u.any():
                        t_yu[b_u] = t_yu[b_u] * 8
                        b_u = f(t_yu) < t_z
                return t_yl,t_yu

        def next_t_vy_alt(t_yl, t_yu, t_vy, my_diff):
                t_yu = t_vy*(my_diff > 0) + t_yu*(my_diff <= 0)
                t_yl = t_vy*(my_diff < 0) + t_yl*(my_diff >= 0)
                t_vy = (t_yu + t_yl) / 2
                return t_vy, t_yu, t_yl
        
        #control whether f_in is invertible:
        assert np.isfinite(alpha).all(), "some alphas are not finite"
        assert (alpha >= 0).all(), "some alphas are negative" 
        assert (w > 0).all(), "some ws are nonpositive"

        #control whether t_Z is out of range:


        N = alpha.shape[0]
        my_shape = [1]*t_z.dim()
        my_shape[0] = N
        t_zu = (alpha.sum(1) + b).reshape(my_shape)
        t_zl = b.reshape(my_shape)
        # if not ((t_z < t_zu).all() and (t_z > t_zl).all()):
        #     print ("ERR: z out of range")
        #     pdb.set_trace()
        
        t_yl, t_yu = initialBounds(t_z, f)

        t_vy = torch.zeros(t_z.shape)            
        for iter in range(max_niter):
                t_vz = f(t_vy)
                my_diff = t_vz-t_z
                if (abs(my_diff)<tol).all():
                        break
                # bin = my_diff > 0
                # bb =  my_diff < 0
                # t_yu[bin] = t_vy[bin]
                # t_yl[bb] = t_vy[bb]
                # t_vy = (t_yu + t_yl) / 2

                t_vy, t_yu, t_yl = next_t_vy_alt(t_yl, t_yu, t_vy, my_diff)        
        #print(iter)
        return t_vy






if __name__ == '__main__':
        M = 5
        T = 10
        N = 12
        t_x = torch.randn(N, T, T, T)
        alpha = torch.rand(N, M)
        w = torch.ones(N,M)
        k = torch.rand(N,M)
        b = torch.ones(N)
        print('computing...')
        t_y, t_y1 = f_param_tensor(t_x, alpha, w, k, b, produce_f_prime=True)
        print(t_y)
        t_y2 = np.zeros(t_x.shape)
        t_y3 = np.zeros(t_x.shape)
        it = np.nditer(t_x, flags=['multi_index'])
        for x in it:
                i = it.multi_index[0]
                t_y2[it.multi_index] = f_param(
                        t_x.numpy()[it.multi_index], i,
                        alpha.numpy(), w.numpy(), k.numpy(), b.numpy())
                t_y3[it.multi_index] = f_prime_param(
                        t_x.numpy()[it.multi_index], i,
                        alpha.numpy(), w.numpy(), k.numpy(), b.numpy())        
        print(t_y2)

        print(np.mean((t_y.numpy()-t_y2)**2))
        print(np.mean((t_y1.numpy()-t_y3)**2))
        
        t_x0 = g_tensor(f_param_tensor, t_y, alpha, w, k, b)
        print(np.mean((t_x.numpy()-t_x0.numpy())**2))