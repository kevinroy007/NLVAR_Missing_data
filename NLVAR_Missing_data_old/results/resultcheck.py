import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import pdb
from sklearn import metrics
from matplotlib import rc,rcParams
from pylab import *

A_true = pickle.load(open("A_true.txt","rb"))
A_true_1 = pickle.load(open("A_true_1_10.txt","rb"))
z_data = pickle.load(open("z_data_woAs.txt","rb")) 
z_data_1 = pickle.load(open("A_wAs_10_4000.txt","rb"))
cost_n = pickle.load(open("indi_results/cost_test_0.0018_.txt","rb"))
cost_n_test = pickle.load(open("indi_results/cost_test_0.0018_.txt","rb"))
cost_linear = pickle.load(open("indi_results/cost_linear_0.0023_.txt","rb"))
cost_linear_test = pickle.load(open("indi_results/cost_linear_test_0.0023_.txt","rb"))
A_n = pickle.load(open("indi_results/A_n_0.0018_.txt","rb"))
A_l = pickle.load(open("indi_results/A_l_0.0035_.txt","rb"))
alpha = pickle.load(open("alpha.txt","rb"))
g = pickle.load(open("g.txt","rb"))
lam = pickle.load(open("../lambda_sweep_f_n_1/lam_LVAR.txt","rb"))
NE = pickle.load(open("NE.txt","rb"))
hat_z_t = pickle.load(open("hat_z_t.txt","rb"))

print(lam)

N,N,P = A_true.shape
N,T  = z_data.shape
N,M = alpha.shape



NE = 100

A_dict = {}

for i in range(len(lam)):

    A_dict[lam[i]] = pickle.load(open("../lambda_sweep_f_n_1/A_n_"+str(np.round(lam[i],5))+"_.txt","rb"))

pdb.set_trace()


rc('axes', linewidth=2)
figure, axis = plt.subplots(1)


legend_prop = {'weight':'bold'}
t1 = np.arange(0,NE,1)+1
axis.plot(t1,(cost_n[0:NE]),label = "NonLinear VAR_train",linewidth=3)
axis.plot(t1,(cost_n_test[0:NE]),label = "NonLinear VAR_test",linewidth=3)
axis.plot(t1,(cost_linear[0:NE]),label = "Linear VAR_train",linewidth=3)
axis.plot(t1,(cost_linear_test[0:NE]),label = "Linear VAR_test",linewidth=3)
axis.set_title("Cost cmparison LinearVAR vs Non LinearVAR")
axis.set_xlabel("Epoch",fontsize=20)
axis.set_ylabel("NMSE",fontsize=20)
axis.grid()
axis.legend(prop={"size":20})
axis.xaxis.set_tick_params(labelsize=20)
axis.yaxis.set_tick_params(labelsize=20)







fig = plt.figure()
#fig.suptitle(" N = "+str(N)+" P = "+str(P)+" T = "+str(T)+ " lambda = "+str(lamda)+" M = "+str(M))
ax1 = fig.add_subplot(3,2,1)
ax1a = fig.add_subplot(3,2,2)


ax2 = fig.add_subplot(3,2,3)
ax2a = fig.add_subplot(3,2,4)


ax3 = fig.add_subplot(3,2,5)
ax3a = fig.add_subplot(3,2,6)


ax1.set_ylabel('Ture Adjacency matrix', fontsize=16)
ax2.set_ylabel('A_l_0.0001', fontsize=16)
ax3.set_ylabel(' Non linear VAR', fontsize=16)

ax1.title.set_text('P = 1')
ax1a.title.set_text('P = 2')




cb1 =  ax1.imshow(A_true_1[:,:,0], vmin=0, vmax=0.427, cmap='jet', aspect='auto')
cb1a =  ax1a.imshow(A_true_1[:,:,1], vmin=0, vmax=0.427, cmap='jet', aspect='auto')


cb2 =  ax2.imshow(A_l[:,:,0], vmin=0, vmax=0.427, cmap='jet', aspect='auto')
cb2a =  ax2a.imshow(A_l[:,:,1], vmin=0, vmax=0.427, cmap='jet', aspect='auto')


cb3 =  ax3.imshow(A_n[:,:,0], vmin=0, vmax=0.427, cmap='jet', aspect='auto')
cb3a =  ax3a.imshow(A_n[:,:,1], vmin=0, vmax=0.427, cmap='jet', aspect='auto')


fig.colorbar(cb1,ax = ax1,orientation='vertical')
fig.colorbar(cb2,ax = ax2,orientation='vertical')
fig.colorbar(cb3,ax = ax3, orientation='vertical')
fig.colorbar(cb1a,ax = ax1a,orientation='vertical')
fig.colorbar(cb2a,ax = ax2a,orientation='vertical')
fig.colorbar(cb3a,ax = ax3a, orientation='vertical')

#remove after meeting
# fig = plt.figure()
# fig.suptitle(" N = "+str(N)+" P = "+str(P)+" T = "+str(T)+ " lambda = "+str(lamda)+" M = "+str(M))

#nx.draw(g)

def roc_curve(y_true, y_prob, thresholds):

    fpr = []
    tpr = []
    y_true = np.ceil(y_true)
    for threshold in thresholds:
         
        y_pred = np.where(y_prob >= threshold, 1, 0)

        fp = np.sum((y_pred == 1) & (y_true == 0))
        tp = np.sum((y_pred == 1) & (y_true == 1))

        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))

        fpr.append(fp / (fp + tn))
        tpr.append(tp / (tp + fn))

    return fpr, tpr


fprP_n = []
tprP_n = []
fprP_l = []
tprP_l = []

thresholds = np.arange(-0,0.7,0.01)

for p in range(P):
    fpr = [0]*thresholds
    tpr = [0]*thresholds

    fpr,tpr = roc_curve(A_true_1[:,:,p],A_n[:,:,p],thresholds)

    fprP_n.append(fpr)
    tprP_n.append(tpr)

for p in range(P):
    fpr = [0]*thresholds
    tpr = [0]*thresholds

    fpr,tpr= roc_curve(A_true_1[:,:,p],A_l[:,:,p],thresholds)

    fprP_l.append(fpr)
    tprP_l.append(tpr)

#pdb.set_trace()

figure, axis = plt.subplots(1, P)
AUC_n = []
AUC_l = []

pdb.set_trace()

for p in range(P):
    
    
    axis[ p].plot(fprP_n[p],tprP_n[p],'-bo', label='Nonlinear_VAR ')
    axis[ p].plot(fprP_l[p],tprP_l[p],'-ro', label='linear_VAR ')
    axis[ p].set_ylabel("tpr")
    axis[ p].set_xlabel("fpr")
    axis[ p].set_title("ROC__ "+str(P))
    # for i2 in range(len(thresholds)):
    #     axis[ p].text(fprP_n[p][i2],tprP_n[p][i2],str(np.round(thresholds[i2],5)))
    axis[ p].legend()

    #axis[ p].legend()
    AUC_n.append(metrics.auc(fprP_n[p], tprP_n[p]))
    AUC_l.append(metrics.auc(fprP_l[p], tprP_l[p]))
# figure.suptitle("Hyperparmeter sweep")

print(AUC_n)
print(AUC_l)

# T1 = np.arange(int(T*0.8),T,1)
# figure, axi = plt.subplots()

# axi.plot(T1,hat_z_t[1,int(T*0.8):T],'-bo', label='linear_VAR ')
# axi.plot(T1,z_data_1[1,int(T*0.8):T],'-ro', label='true_signal ')
# axi.set_ylabel("sensor measurement")
# axi.set_xlabel("time stamps")
# axi.set_title("ROC__ "+str(P))
# axi.legend()

plt.show()
