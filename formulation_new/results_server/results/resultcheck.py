import sys
sys.path.append('indi_results')
from resultcheck_l import optimum_lam_1
from resultcheck_n import optimum_lam
import pickle
import numpy as np
#import networkx as nx
import matplotlib.pyplot as plt
import pdb
from sklearn import metrics
from matplotlib import rc,rcParams
from pylab import *
 
lam_l,lam_l_s = optimum_lam_1()
lam_n,lam_n_s = optimum_lam()


lam_l = np.round(lam_l,7)
lam_n = np.round(lam_n,7)
lam_l_s = np.round(lam_l_s,7)
lam_n_s = np.round(lam_n_s,7)

print("optimum lambda linear",lam_l)
print("optimum lambda nonlinear",lam_n)

print("optimum lambda linear sliced",lam_l_s)
print("optimum lambda nonlinear sliced",lam_n_s)


A_true_1 = pickle.load(open("A_true_1_10.txt","rb"))

cost_n = pickle.load(open("../lambda_sweep_f_n/cost_"+str(lam_n)+"_.txt","rb"))
cost_n_test = pickle.load(open("../lambda_sweep_f_n/cost_test_"+str(lam_n)+"_.txt","rb"))
cost_linear = pickle.load(open("../lambda_sweep_f/cost_linear_"+str(lam_l)+"_.txt","rb"))
cost_linear_test = pickle.load(open("../lambda_sweep_f/cost_linear_test_"+str(lam_l)+"_.txt","rb"))

cost_n_s = pickle.load(open("../lambda_sweep_f_n/cost_"+str(lam_n)+"_s.txt","rb"))
cost_n_test_s = pickle.load(open("../lambda_sweep_f_n/cost_test_"+str(lam_n)+"_s.txt","rb"))
cost_linear_s = pickle.load(open("../lambda_sweep_f/cost_linear_"+str(lam_l)+"_s.txt","rb"))
cost_linear_test_s = pickle.load(open("../lambda_sweep_f/cost_linear_test_"+str(lam_l)+"_s.txt","rb"))


A_n = pickle.load(open("../lambda_sweep_f_n/A_n_"+str(lam_n)+"_.txt","rb"))
A_l = pickle.load(open("../lambda_sweep_f/A_l_"+str(lam_l)+"_.txt","rb"))

A_n_s = pickle.load(open("../lambda_sweep_f_n/A_n_"+str(lam_n)+"_s.txt","rb"))
A_l_s = pickle.load(open("../lambda_sweep_f/A_l_"+str(lam_l)+"_s.txt","rb"))


input_data_filename = "A_wAs_10_fun_3_n.txt"
z_data = pickle.load(open(input_data_filename,"rb"))
NE = 20000
#print(lamda)

N,N,P = A_true_1.shape
N,T  = z_data.shape
#N,M = alpha.shape




#pdb.set_trace()
#NE = 100

rc('axes', linewidth=2)
figure, axis = plt.subplots(1)


legend_prop = {'weight':'bold'}
t1 = np.arange(0,NE,1)+1
axis.plot(t1,(cost_n[0:NE]),label = "NonLinear VAR_train",linewidth=3)
axis.plot(t1,(cost_n_test[0:NE]),label = "NonLinear VAR_test",linewidth=3)
axis.plot(t1,(cost_linear[0:NE]),label = "Linear VAR_train",linewidth=3)
axis.plot(t1,(cost_linear_test[0:NE]),label = "Linear VAR_test",linewidth=3)

axis.plot(t1,(cost_n_s[0:NE]),label = "NonLinear VAR_train sliced",linewidth=3)
axis.plot(t1,(cost_n_test_s[0:NE]),label = "NonLinear VAR_test sliced",linewidth=3)
axis.plot(t1,(cost_linear_s[0:NE]),label = "Linear VAR_train sliced",linewidth=3)
axis.plot(t1,(cost_linear_test_s[0:NE]),label = "Linear VAR_test sliced",linewidth=3)
axis.set_ylim([0, 1])
axis.set_title("Cost cmparison LinearVAR vs Non LinearVAR")
axis.set_xlabel("Epoch",fontsize=20)
axis.set_ylabel("NMSE",fontsize=20)
axis.grid()
axis.legend(prop={"size":10})
axis.xaxis.set_tick_params(labelsize=20)
axis.yaxis.set_tick_params(labelsize=20)





fig = plt.figure()
#fig.suptitle(" N = "+str(N)+" P = "+str(P)+" T = "+str(T)+ " lambda = "+str(lamda)+" M = "+str(M))
ax1 = fig.add_subplot(5,2,1)
ax1a = fig.add_subplot(5,2,2)


ax2 = fig.add_subplot(5,2,3)
ax2a = fig.add_subplot(5,2,4)


ax3 = fig.add_subplot(5,2,5)
ax3a = fig.add_subplot(5,2,6)


ax4 = fig.add_subplot(5,2,7)
ax4a = fig.add_subplot(5,2,8)

ax5 = fig.add_subplot(5,2,9)
ax5a = fig.add_subplot(5,2,10)


ax1.set_ylabel('Ture Adjacency matrix', fontsize=7)
ax2.set_ylabel('linear', fontsize=7)
ax3.set_ylabel(' Non linear VAR', fontsize=7)
ax4.set_ylabel(' linear VAR_sliced', fontsize=7)
ax5.set_ylabel(' Non linear VAR_sliced', fontsize=7)

ax1.title.set_text('P = 1')
ax1a.title.set_text('P = 2')




cb1 =  ax1.imshow(A_true_1[:,:,0], vmin=0, vmax=0.427, cmap='jet', aspect='auto')
cb1a =  ax1a.imshow(A_true_1[:,:,1], vmin=0, vmax=0.427, cmap='jet', aspect='auto')


cb2 =  ax2.imshow(A_l[:,:,0], vmin=0, vmax=0.427, cmap='jet', aspect='auto')
cb2a =  ax2a.imshow(A_l[:,:,1], vmin=0, vmax=0.427, cmap='jet', aspect='auto')


cb3 =  ax3.imshow(A_n[:,:,0], vmin=0, vmax=0.427, cmap='jet', aspect='auto')
cb3a =  ax3a.imshow(A_n[:,:,1], vmin=0, vmax=0.427, cmap='jet', aspect='auto')

cb4 =  ax4.imshow(A_l_s[:,:,0], vmin=0, vmax=0.427, cmap='jet', aspect='auto')
cb4a =  ax4a.imshow(A_l_s[:,:,1], vmin=0, vmax=0.427, cmap='jet', aspect='auto')

cb5 =  ax5.imshow(A_n_s[:,:,0], vmin=0, vmax=0.427, cmap='jet', aspect='auto')
cb5a =  ax5a.imshow(A_n_s[:,:,1], vmin=0, vmax=0.427, cmap='jet', aspect='auto')


fig.colorbar(cb1,ax = ax1,orientation='vertical')
fig.colorbar(cb2,ax = ax2,orientation='vertical')
fig.colorbar(cb3,ax = ax3, orientation='vertical')
fig.colorbar(cb4,ax = ax4, orientation='vertical')
fig.colorbar(cb5,ax = ax5, orientation='vertical')

fig.colorbar(cb1a,ax = ax1a,orientation='vertical')
fig.colorbar(cb2a,ax = ax2a,orientation='vertical')
fig.colorbar(cb3a,ax = ax3a, orientation='vertical')
fig.colorbar(cb4a,ax = ax4a, orientation='vertical')
fig.colorbar(cb5a,ax = ax5a, orientation='vertical')

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

#pdb.set_trace()

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
