
import numpy as np
import pdb
import pickle
import matplotlib.pyplot as plt
from sklearn import metrics

def roc_curve(y_true, y_pred):

    fpr = 0
    tpr = 0


    fp = np.sum((y_pred == 1) & (y_true == 0))
    tp = np.sum((y_pred == 1) & (y_true == 1))

    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))

    fpr = (fp / (fp + tn))
    tpr = (tp / (tp + fn))


    return fpr, tpr




#lam = pickle.load(open("VAR_solid_result/sparse_nonlinear_var_P4N5cbrt_1000_solid_result_l2/results/lam_NLVAR.txt","rb"))
#lamn = np.arange(90,100,1)

lam1 = pickle.load(open("lambda_sweep_10/lam_LVAR.txt","rb"))
lam2 = np.arange(0.1,0.5,0.01)
lam3 = np.arange(1,20,1)
lam4 = np.arange(20,500,50)

lam = np.append(lam1,lam2)
lam = np.append(lam,lam3)
lam = np.append(lam,lam4)

lam.sort()


fpr  = [0]*lam
tpr  = [0]*lam


pdb.set_trace()
    
GC_true = pickle.load(open("results/A_true_1_10.txt","rb"))
GC_true = np.ceil(GC_true)



for i2 in range(len(lam)):
    
    GC_est =  pickle.load(open("lambda_sweep_10/A_l_"+str(np.round(lam[i2],4))+"_.txt","rb"))
    
    GC_est = np.ceil(GC_est)
    #pdb.set_trace()


    fpr[i2],tpr[i2] = roc_curve(GC_true[:,:,0],GC_est[:,:,0])

        

#pdb.set_trace()  


# pickle.dump(fpr,open("fpr_tpr_united/fpr_nlvar.txt","wb"))
# pickle.dump(tpr,open("fpr_tpr_united/tpr_nlvar.txt","wb"))

fpr.sort()
tpr.sort()

figure, axis = plt.subplots(1)

AUC = []


axis.plot(fpr,tpr, '-bo', label='lambda ')
axis.set_ylabel("tpr")
axis.set_xlabel("fpr")
axis.set_title("ROC_LVAR_ ")
for i2 in range(len(lam)):
        axis.text(fpr[i2],tpr[i2],str(np.round(lam[i2],5)))
axis.legend()



AUC.append(metrics.auc(fpr, tpr))
    
    
# figure.suptitle("Hyperparmeter sweep")

#pdb.set_trace()

print(AUC)

plt.show()







