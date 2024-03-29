import numpy as np
import pdb
from f import f_param, sigmoid


def g_bisection(z, i, alpha, w, k, b):
    
   
    # def sigmoid(x):
      
      

    #  if x.any()<0:
    #     sig =  np.exp(x) / (1+np.exp(x))
    #  else:
    #     sig = 1 / (1 + np.exp(-x))

    #  return sig 

    
    # def sigmoid(x):

    #     try:
            
    #         return np.where(x >= 0, 
    #             1 / (1 + np.exp(-x)), 
    #             np.exp(x) / (1 + np.exp(x)))

    #     except: pdb.set_trace()
        
    # def f(x,i): 
    #     a3 = 0
    #     a3 = alpha[i,:]* sigmoid(w[i,:]*x-k[i,:])
    #     return (a3.sum() +b[i])
    def f(x,i):
        return f_param(x, i, alpha, w, k, b)

    assert np.isfinite(alpha).all(), "some alphas are not finite"#
    assert (alpha >= 0).all(), "some alphas are negative" 
    assert (w > 0).all(), "some ws are nonpositive" 

    max_niter = 1000
    zu = np.sum(alpha[i,:]) + b[i]
    zl = b[i]
    vy = 0  

    # if z >= zu or z <= zl:
    #     pdb.set_trace()

    #assert z < zu and z > zl,"z out of range"

    yl = -10
    while f(yl,i) > z:
        yl = yl*10
    yu = 10
    while f(yu,i) < z:
        yu = yu*10
    for iter in range(max_niter): 
        vz = f(vy,i) 
        if vz > z:
            yu = vy
        else:
            yl = vy
        #bisection iteration:                         
        vy = (yu + yl) / 2
        if abs(vz-z)<1e-6: #stopping criterion
            break
    return vy
