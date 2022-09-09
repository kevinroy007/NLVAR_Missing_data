
import numpy as np
import warnings, pdb
import torch
from matplotlib import rc,rcParams
import matplotlib.pyplot as plt
import pickle

NE = 20

t1 = np.arange(0,NE,1)+1

#cost_30 = pickle.load(open("cost_30_.txt","rb"))
cost_20 = pickle.load(open("cost_20_.txt","rb"))
cost_10 = pickle.load(open("cost_10_.txt","rb"))
figure, axis = plt.subplots(1)
#axis.plot(t1[0:30],cost_30[0:30],'-r',label = "cost 30 epochs ",linewidth=3)
axis.plot(t1[0:20],cost_20[0:20],'-r',label = "cost 20 epochs ",linewidth=3)
axis.plot(t1[0:10],cost_10,'-b',label = "cost 10 epochs ",linewidth=3)
axis.set_xlabel("epoch",fontsize=20)
axis.set_ylabel("NMSE", fontsize=20)
axis.grid()
axis.legend(prop={"size":20})
axis.xaxis.set_tick_params(labelsize=20)
axis.yaxis.set_tick_params(labelsize=20)
plt.show()