import numpy as np
import pdb

def sigmoid(x):

        try:
            
            return np.where(x >= 0, 
                1 / (1 + np.exp(-x)), 
                np.exp(x) / (1 + np.exp(x)))

        except: pdb.set_trace()

gamma = 1e-3


def f_param(x,i, alpha, w, k, b): 
    a3 = alpha[i,:]* sigmoid(w[i,:]*x-k[i,:]) 
    return a3.sum() +b[i]+ gamma*x

def f_prime_param(x,i, alpha, w, k, b):
    a = alpha[i,:] * sigmoid(w[i,:]*x-k[i,:]) * (1-sigmoid(w[i,:]*x-k[i,:]))*(w[i,:]) 
    return a.sum() + gamma



