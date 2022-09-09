import sys
sys.path.insert(0, '/Users/kevinroy/Documents/PhD/github/NLVAR_proximal_descent/NLVAR_proximal_descent/formulation_A')

import numpy as np
from matplotlib import rc,rcParams
import matplotlib.pyplot as plt
import pickle
from basic_functions import f_param

N = 10
P =2
M =10



alpha= pickle.load(open("alpha.pickle","rb"))
w = pickle.load(open("w.pickle","rb"))
k = pickle.load(open("k.pickle","rb")) 
b = pickle.load(open("b.pickle","rb"))
A = pickle.load(open("A.txt","rb"))


t1 = np.arange(-1,1,0.001)
z  = np.zeros((len(t1)))
for t in range(len(t1)):

    z[t] = f_param(t1[t],9, alpha, w, k, b)




figure, axis = plt.subplots(1)
rc('axes', linewidth=2)
axis.plot(t1,z,'-bo',label = "function",linewidth=3)

axis.set_xlabel("y",fontsize=20)
axis.set_ylabel("z",fontsize=20)
axis.grid()
axis.legend(prop={"size":20})
axis.xaxis.set_tick_params(labelsize=20)
axis.yaxis.set_tick_params(labelsize=20)
    
#     axis[n].set_ylabel("z")
#     axis[n].set_xlabel("y")
#     #axis[n].set_title("ROC__ "+str(P))
#     # for i2 in range(len(thresholds)):
#     #     axis[ p].text(fprP_n[p][i2],tprP_n[p][i2],str(np.round(thresholds[i2],5)))
#     axis[n].legend()
# for n in range(N):
    
#     y = f_param(t1,n, alpha, w, k, b)
#     axis[n].plot(t1,y,'-bo',label = "function "+str(n),linewidth=3)
    
#     axis[n].set_ylabel("z")
#     axis[n].set_xlabel("y")
#     #axis[n].set_title("ROC__ "+str(P))
#     # for i2 in range(len(thresholds)):
#     #     axis[ p].text(fprP_n[p][i2],tprP_n[p][i2],str(np.round(thresholds[i2],5)))
#     axis[n].legend()









plt.show()