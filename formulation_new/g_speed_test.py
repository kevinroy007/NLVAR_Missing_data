
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pdb, time
from pynverse import inversefunc
from matplotlib import rc

from g_bisection import g_bisection as g_b
from basic_functions import sigmoid

# here y takes a range of values 
def g_pyinverse(y,i, alpha, w, k, b):
    
    def g(x):
        return f_param(x,i, alpha, w, k, b)

    y_inv = inversefunc(g,y)
    
    return y_inv

def f_param(x,i, alpha, w, k, b): 
    a3 = alpha[i,:]* sigmoid(w[i,:]*x-k[i,:]) 
    return a3.sum() +b[i]


if __name__ == '__main__':

    alpha = pickle.load(open("function_para/alpha.txt","rb"))
    w = pickle.load(open("function_para/w.txt","rb"))
    k = pickle.load(open("function_para/k.txt","rb")) 
    b = pickle.load(open("function_para/b.txt","rb"))
    #gamma = 0.25*alpha[:,0]*w[:,0]
    i = 1
   
   
    z = np.arange(-3,3,0.003)
    y = np.ones((len(z)))

    for t in range(len(z)):
        y[t] = f_param(z[t],i, alpha, w, k, b)

    ####################################################
    #  comparison starts here
    T1 = time.time()
    y_inv = np.ones((len(y)))

    for t in range(len(y)):
        y_inv[t] = g_pyinverse(y[t],i, alpha, w, k, b)
    T2 = time.time()
    ######################################################
    y_inv_old = np.ones((len(y)))

    for t in range(len(y)):
        y_inv_old[t] = g_b(y[t],i, alpha, w, k, b)

    T3 = time.time()

    #####################################################
    print("time taken for g new with pyinverse",T2-T1)
    print("time taken for g_bisection"         ,T3-T2)

    rc('axes', linewidth=2)

    # plotting function q
    figure, axis = plt.subplots(1)
    axis.plot(z,y,'-bo',label = "function f_param ",linewidth=3)
    axis.set_xlabel("z",fontsize=20)
    axis.set_ylabel("y",fontsize=20)
    axis.grid()
    axis.legend(prop={"size":20})
    axis.xaxis.set_tick_params(labelsize=20)
    axis.yaxis.set_tick_params(labelsize=20)

    # plotting function q inverse
    figure, axis = plt.subplots(1)
    axis.plot(y,y_inv,'-bo',label = "function q_inverse with pyinverse", \
        linewidth=3)
    axis.set_xlabel("y",fontsize=20)
    axis.set_ylabel("y_inv",fontsize=20)
    axis.grid()
    axis.legend(prop={"size":20})
    axis.xaxis.set_tick_params(labelsize=20)
    axis.yaxis.set_tick_params(labelsize=20)

    figure, axis = plt.subplots(1)
    axis.plot(y,y_inv,'-bo',label = "function q_inverse with g_bisection", \
        linewidth=3)
    axis.set_xlabel("y",fontsize=20)
    axis.set_ylabel("y_inv",fontsize=20)
    axis.grid()
    axis.legend(prop={"size":20})
    axis.xaxis.set_tick_params(labelsize=20)
    axis.yaxis.set_tick_params(labelsize=20)

    plt.show()



