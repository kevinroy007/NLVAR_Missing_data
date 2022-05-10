import numpy as np
import pickle
import pdb
import matplotlib.pyplot as plt
from matplotlib import rc


A_n = pickle.load(open("A_n_0.0001_.txt","rb"))
A_true_1 = pickle.load(open("A_true_1_10.txt","rb"))

N,N,P = A_n.shape

thresholds = np.arange(-0,0.7,0.01)




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


def stack_tensor(x):

    A_n_stack = x[:,:,0]
    for p in range(1,P):

        A_n_stack = np.concatenate((A_n_stack,x[:,:,p]))

    return A_n_stack

fig = plt.figure()


A_true_s = stack_tensor(A_true_1)
A_n_s = stack_tensor(A_n)

pdb.set_trace()

######################################################

fpr,tpr = roc_curve(A_true_1,A_n,thresholds)



##################################################






rc('axes', linewidth=2)
figure, axis = plt.subplots(1)


legend_prop = {'weight':'bold'}

axis.plot(fpr,tpr,'-bo',label = "NonLinear",linewidth=3)
#axis.plot(t1,(cost_n_test[0:NE]),label = "NonLinear VAR_test",linewidth=3)
#axis.plot(t1,(cost_linear[0:NE]),label = "Linear VAR_train",linewidth=3)
#axis.plot(t1,(cost_linear_test[0:NE]),label = "Linear VAR_test",linewidth=3)
axis.set_title("Cost cmparison LinearVAR vs Non LinearVAR")
axis.set_xlabel("Epoch",fontsize=20)
axis.set_ylabel("NMSE",fontsize=20)
axis.grid()
axis.legend(prop={"size":20})
axis.xaxis.set_tick_params(labelsize=20)
axis.yaxis.set_tick_params(labelsize=20)

plt.show()