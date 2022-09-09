import sys
sys.path.append('indi_results')
from resultcheck_l import optimum_lam_linear
from resultcheck_n import optimum_lam_nonlinear
import pickle
import numpy as np
#import networkx as nx
import matplotlib.pyplot as plt
import pdb
from sklearn import metrics
from matplotlib import rc,rcParams
from pylab import *
 
#lam_l = optimum_lam_linear()
lam_n = optimum_lam_nonlinear()

#lam_l = np.round(lam_l,7)
lam_n = np.round(lam_n,7)

#print("optimum lambda linear",lam_l)
print("optimum lambda nonlinear",lam_n)


A_true_1 = pickle.load(open("../data/synthetic/A_true_P_2_T3000.pickle","rb"))

cost_n = pickle.load(open("../lambda_sweep_f_n_0.10/cost_"+str(lam_n)+"_.txt","rb"))
cost_n_test = pickle.load(open("../lambda_sweep_f_n_0.10/cost_test_"+str(lam_n)+"_.txt","rb"))

# cost_n_2 = pickle.load(open("../lambda_sweep_f_n_0.10/cost_"+str(lam_n)+"_.txt","rb"))
# cost_n_test_2 = pickle.load(open("../lambda_sweep_f_n_0.10/cost_test_"+str(lam_n)+"_.txt","rb"))

A_n = pickle.load(open("../lambda_sweep_f_n_0.10/A_n_"+str(lam_n)+"_.txt","rb"))

NE = pickle.load(open("../lambda_sweep_f_n_0.10/NE.txt","rb"))
#print(lamda)

N,N,P = A_true_1.shape


rc('axes', linewidth=2)
figure, axis = plt.subplots(1)


pdb.set_trace()

legend_prop = {'weight':'bold'}
t1 = np.arange(0,NE,1)+1
axis.plot(t1,(cost_n[0:NE]),label = "NonLinear VAR_train",linewidth=3)
axis.plot(t1,(cost_n_test[0:NE]),label = "NonLinear VAR_test",linewidth=3)
axis.set_ylim([0, 1])
axis.set_title("NMSE Non LinearVAR")
axis.set_xlabel("Epoch",fontsize=20)
axis.set_ylabel("NMSE",fontsize=20)
axis.grid()
axis.legend(prop={"size":20})
axis.xaxis.set_tick_params(labelsize=20)
axis.yaxis.set_tick_params(labelsize=20)




plt.show()
