import sys
import os
from tkinter.ttk import Sizegrip

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
from mpl_toolkits.axes_grid1 import make_axes_locatable
os.system("clear")
# file_directory = sys.path[0]
# os.chdir(file_directory)

folder_num_1 = 0.05
folder_num_2 = format(0.3, '.1f')
folder_num_3 = 0.5


results_folder_loc_1 = "../lambda_sweep_f_n_"+str(folder_num_1)+"/"


z_tilde_data = pickle.load(open("z_tilde_data.txt","rb"))

lam_1 = optimum_lam(results_folder_loc_1)


lam_1 = np.round(lam_1,7)


print("optimum lambda 5 % missing",lam_1)


A_true_1 = pickle.load(open("../data/synthetic/A_true_P_2_T3000.pickle","rb"))

cost_n_1 = pickle.load(open(results_folder_loc_1 + "cost_"+str(lam_1)+"_.txt","rb"))
cost_n_1_t = pickle.load(open(results_folder_loc_1 + "cost_test_"+str(lam_1)+"_.txt","rb"))



A_n_1 = pickle.load(open(results_folder_loc_1 + "A_n_"+str(lam_1)+"_.txt","rb"))


input_data_filename = "../data/synthetic/synthetic_dataset_P2_T3000.pickle"
z_data = pickle.load(open(input_data_filename,"rb"))
NE = pickle.load(open(results_folder_loc_1 + "NE.txt","rb"))
 
z_1 = pickle.load(open(results_folder_loc_1 + "z_learned"+str(lam_1)+"_.txt","rb"))



#print(lamda)

N,N,P = A_true_1.shape
T  = z_1[1].shape
#N,M = alpha.shape





#pdb.set_trace()
#NE = 100

rc('axes', linewidth=2)
figure, axis = plt.subplots(1)



legend_prop = {'weight':'bold'}
t1 = np.arange(0,NE,1)+1
axis.plot(t1,(cost_n_1),"-bo",markersize=9,label = "NLVAR_train 5 % missing",linewidth=3)
axis.plot(t1,(cost_n_1_t),"-y*",markersize=9,label = "NLVAR_test 5 % missing",linewidth=3)

axis.set_ylim([0, 1])
#axis.set_title("Cost cmparison LinearVAR vs Non LinearVAR")
axis.set_xlabel("Epoch",fontsize=35)
axis.set_ylabel("NMSE",fontsize=35)
axis.grid()
axis.legend(prop={"size":30})
axis.xaxis.set_tick_params(labelsize=30)
axis.yaxis.set_tick_params(labelsize=30)



fig = plt.figure()
#fig.suptitle(" N = "+str(N)+" P = "+str(P)+" T = "+str(T)+ " lambda = "+str(lamda)+" M = "+str(M))
ax1 = fig.add_subplot(3,2,1)
ax1a = fig.add_subplot(3,2,2)


ax2 = fig.add_subplot(3,2,3)
ax2a = fig.add_subplot(3,2,4)


ax3 = fig.add_subplot(3,2,5)
ax3a = fig.add_subplot(3,2,6)


ax1.set_ylabel('True Adjacency', fontsize=30)
ax2.set_ylabel('5 % missing', fontsize=30)
ax3.set_ylabel('50 % missing', fontsize=30)


ax1.title.set_text('P = 1')
ax1a.title.set_text('P = 2')
ax1.title.set_size(30)
ax1a.title.set_size(30)


ax1.tick_params(axis='both', which='major', labelsize=30)
ax1a.tick_params(axis='both', which='major', labelsize=30)
ax2.tick_params(axis='both', which='major', labelsize=30)
ax2a.tick_params(axis='both', which='major', labelsize=30)
ax3.tick_params(axis='both', which='major', labelsize=30)
ax3a.tick_params(axis='both', which='major', labelsize=30)


cb1 =  ax1.imshow(A_true_1[:,:,0], vmin=0, vmax=0.427, cmap='jet', aspect='auto')
cb1a =  ax1a.imshow(A_true_1[:,:,1], vmin=0, vmax=0.427, cmap='jet', aspect='auto')



cb2 =  ax2.imshow(A_n_1[:,:,0], vmin=0, vmax=0.427, cmap='jet', aspect='auto')
cb2a =  ax2a.imshow(A_n_1[:,:,1], vmin=0, vmax=0.427, cmap='jet', aspect='auto')



fig.colorbar(cb1,ax = ax1,orientation='vertical')
fig.colorbar(cb2,ax = ax2,orientation='vertical')
fig.colorbar(cb1a,ax = ax1a,orientation='vertical')
fig.colorbar(cb2a,ax = ax2a,orientation='vertical')



def roc_curve(y_true, y_prob, thresholds):

    fpr = []
    tpr = []
    y_true_bin = np.where(abs(y_true)>0, 1, 0)

    for threshold in thresholds:
         
        y_pred = np.where(np.abs(y_prob) >= threshold, 1, 0)

        fp = np.sum((y_pred == 1) & (y_true_bin == 0))
        tp = np.sum((y_pred == 1) & (y_true_bin == 1))

        fn = np.sum((y_pred == 0) & (y_true_bin == 1))
        tn = np.sum((y_pred == 0) & (y_true_bin == 0))

        fpr.append(fp / (fp + tn))
        tpr.append(tp / (tp + fn))

    return fpr, tpr


fprP_1 = []
tprP_1 = []

fprP_2 = []
tprP_2 = []

fprP_3 = []
tprP_3 = []

thresholds = np.arange(0,0.5,0.001)

for p in range(P):
    fpr = [0]*thresholds
    tpr = [0]*thresholds

    fpr,tpr = roc_curve(A_true_1[:,:,p],A_n_1[:,:,p],thresholds)

    fprP_1.append(fpr)
    tprP_1.append(tpr)



#pdb.set_trace()

figure, axis = plt.subplots(1, P)
AUC_1 = []
AUC_2 = []
AUC_3 = []
#pdb.set_trace()

for p in range(P):
    
    
    axis[ p].plot(fprP_1[p],tprP_1[p],'-bo', label='NLVAR 5 % missing ')
   
    axis[ p].set_ylabel("tpr")
    axis[ p].set_xlabel("fpr")
    axis[ p].set_title("ROC lag p = "+str(p+1))
    # for i2 in range(len(thresholds)):
    #     axis[ p].text(fprP_n[p][i2],tprP_n[p][i2],str(np.round(thresholds[i2],5)))
    axis[ p].legend()

    #axis[ p].legend()
    AUC_1.append(metrics.auc(fprP_1[p], tprP_1[p]))
    

# figure.suptitle("Hyperparmeter sweep")

print(AUC_1)
print(AUC_2)
print(AUC_3)


figure, axi = plt.subplots(1,1)
T = np.arange(0,1000,1)

axi.plot(T[:],z_1[0,0:1000],'ro',mfc='none', label='reconstruction using NLVAR for 5 % missing data')
axi.plot(T[:],z_data[0,0:1000], label='true_signal ')
axi.plot(T[:],z_tilde_data[0,0:1000], 'xg', label='noisy signal ')

axi.set_ylabel("sensor measurement")
axi.set_xlabel("time stamps")
axi.set_title("ROC ag P = "+str(P))
axi.legend()

ISE_1 = np.square(z_data[0,0:1000]-z_1[0,0:1000])

figure, axi = plt.subplots(1,1)

axi.plot(T[:],ISE_1[:], label='5 % missing_VAR ')


axi.set_ylabel("ISE")
axi.set_xlabel("time stamps")
axi.set_title("ROC__ "+str(P))
axi.legend()

# try to plot the instantaneos error and look rohans paper regarding signal reconstruction

plt.show()
