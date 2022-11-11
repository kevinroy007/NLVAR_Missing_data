import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import pdb
from sklearn import metrics
from matplotlib import rc,rcParams
from pylab import *

def optimum_lam(results_folder_loc_n):
    # cost_n = pickle.load(open("cost_n_proxy2.txt","rb"))
    # cost_n_test = pickle.load(open("cost_n_test_proxy2.txt","rb"))

    #lam = np.arange(0.0001,0.001,0.0001)
    # lam2 = np.arange(0.001,0.05,0.002)
    # lam = np.append(lam1,lam2)
    #NE =  pickle.load(open("NE.txt","rb"))

    lam =  pickle.load(open(results_folder_loc_n + "lam_LVAR.txt","rb"))

  

    

    cost_val = [0]*len(lam)

    #C = pickle.load(open("val_lambda_"+str(lam[0])+"_.txt","rb"))
    # for i in range (lam):
    #     cost_linear = pickle.load(open("cost_linear_"+str(lam[i])+"_.txt","rb"))
    #     cost_linear_val = pickle.load(open("cost_linear_val_"+str(lam[i])+"_.txt","rb"))
    #     cost_linear_test = pickle.load(open("cost_linear_test_"+str(lam[i])+"_.txt","rb"))
    #am = np.arange(0.0015,0.0025,0.0001)
    #pdb.set_trace()
    cost_val = [0]*len(lam)

    for i in range(len(lam)):
        
        cost_val[i] = pickle.load(open(results_folder_loc_n + "val_lambda_"+str(np.round(lam[i],5))+"_.txt","rb"))

    minpos = cost_val.index(min(cost_val))
    optimum_lam = lam[minpos]
 


    rc('axes', linewidth=2)
    figure, axis = plt.subplots(1)

    legend_prop = {'weight':'bold'}
    legend_prop = {'weight':'bold'}


    #t1 = np.arange(0,lam,1)+1
    axis.plot(lam,cost_val,'-bo',label = "NLVAR_val",linewidth=3)
    #axis[ i].plot(val,lam[i], '-bo', label='lambda ')
    #axis.plot(t1,cost_n_test[5:NE],label = "NonLinear VAR_test",linewidth=3)

    #axis.plot(t1,cost_linear[5:NE],label = "Linear VAR_train",linewidth=3)
    #axis.plot(t1,cost_linear_test[5:NE],label = "Linear VAR_test",linewidth=3)
    #axis.set_title("Cost cmparison LinearVAR vs Non LinearVAR")
    axis.set_xlabel("lambda",fontsize=20)
    axis.set_ylabel("validation_error",fontsize=20)
    axis.grid()
    axis.legend(prop={"size":20})
    axis.xaxis.set_tick_params(labelsize=20)
    axis.yaxis.set_tick_params(labelsize=20)



    # t1 = np.arange(NE)+1
    # axis[1, 1].plot(t1,cost_n,label = "NonLinear VAR")
    # axis[1, 1].set_title("Cost magnification of Non LinearVAR")
    # axis[1, 1].set_xlabel("Epoch")
    # axis[1, 1].legend()

    # figure, axis = plt.subplots(1)

    # for i in range(len(lam)):
        
    #     # fpr[i].sort()
    #     # tpr[i].sort()
    #     val = pickle.load(open("val_lambda_"+str(lam[i])+"_.txt","rb"))
    #     axis[ i].plot(val,lam[i], '-bo', label='lambda ')
    #     axis[ i].set_ylabel("validation error")
    #     axis[ i].set_xlabel("lamda")
    #     axis[ i].set_title("Hyperparameter_optimization")

    #     axis[ i].text(val,lam[i],str(np.round(lam[i],5)))
    #     axis[ i].legend()
        

        
    #plt.show()
    return optimum_lam






#remove after meeting
# fig = plt.figure()
# fig.suptitle(" N = "+str(N)+" P = "+str(P)+" T = "+str(T)+ " lambda = "+str(lamda)+" M = "+str(M))

#nx.draw(g)


