import numpy as np
import pdb
from basic_functions import *


def g_bisection(z, i, alpha, w, k, b, tol = 1e-6):
        
    def f(x,i): 
        a3 = alpha[i,:]* sigmoid(w[i,:]*x-k[i,:])
        return a3.sum() +b[i]
    assert np.isfinite(alpha).all(), "some alphas are not finite"#
    assert (alpha >= 0).all(), "some alphas are negative" 
    #pdb.set_trace()
    assert (w > 0).all(), "some ws are nonpositive" 

    max_niter = 1000
    zu = np.sum(alpha[i,:]) + b[i]
    zl = b[i]
    vy = 0   
    if z >= zu or z <= zl:
        pdb.set_trace()
    assert z < zu and z > zl,"z out of range"
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
        if abs(vz-z)<tol: #stopping criterion
            break
    return vy