import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import pdb
from sklearn import metrics
from matplotlib import rc,rcParams
from pylab import *

def optimum_lam_nonlinear():
    # cost_n = pickle.load(open("cost_n_proxy2.txt","rb"))
    # cost_n_test = pickle.load(open("cost_n_test_proxy2.txt","rb"))


    lam =  pickle.load(open("../lambda_sweep_f_n_0.10/lam_LVAR.txt","rb"))
    cost_val = [0]*len(lam)

    cost_val = [0]*len(lam)

    for i in range(len(lam)):
        
        cost_val[i] = pickle.load(open("../lambda_sweep_f_n_0.10/val_lambda_"+str(np.round(lam[i],5))+"_.txt","rb"))

    minpos = cost_val.index(min(cost_val))
    optimum_lam = lam[minpos]
 


    rc('axes', linewidth=2)
    figure, axis = plt.subplots(1)

    legend_prop = {'weight':'bold'}
    legend_prop = {'weight':'bold'}


    #t1 = np.arange(0,lam,1)+1
    axis.plot(lam,cost_val,'-bo',label = "Non linear VAR_val",linewidth=3)
    #axis[ i].plot(val,lam[i], '-bo', label='lambda ')
    #axis.plot(t1,cost_n_test[5:NE],label = "NonLinear VAR_test",linewidth=3)

    
    axis.set_xlabel("lambda",fontsize=20)
    axis.set_ylabel("validation_error",fontsize=20)
    axis.grid()
    axis.legend(prop={"size":20})
    axis.xaxis.set_tick_params(labelsize=20)
    axis.yaxis.set_tick_params(labelsize=20)



    
    return optimum_lam






#remove after meeting
# fig = plt.figure()
# fig.suptitle(" N = "+str(N)+" P = "+str(P)+" T = "+str(T)+ " lambda = "+str(lamda)+" M = "+str(M))

#nx.draw(g)


