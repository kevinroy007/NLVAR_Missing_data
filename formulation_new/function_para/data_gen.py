import numpy as np
from matplotlib import rc,rcParams
import matplotlib.pyplot as plt
import pickle


N = 10
P =2
M =10

def f(x,i): 


        a = b[i]
        for m in range(M):
            a = a + alpha[i][m] * sigmoid(w[i][m]*x-k[i][m])
  
        return a
def sigmoid(l):
        
        return  1/(1+np.exp(-l)) 



alpha= pickle.load(open("alpha.txt","rb"))
w = pickle.load(open("w.txt","rb"))
k = pickle.load(open("k.txt","rb")) 
b = pickle.load(open("b.txt","rb"))
A = pickle.load(open("A.txt","rb"))

t1 = np.arange(-1,1,0.001)

y = f(t1,1)

rc('axes', linewidth=2)
figure, axis = plt.subplots(1)
axis.plot(t1,y,'-bo',label = "Linear VAR_val",linewidth=3)

axis.set_xlabel("lambda",fontsize=20)
axis.set_ylabel("validation_error",fontsize=20)
axis.grid()
axis.legend(prop={"size":20})
axis.xaxis.set_tick_params(labelsize=20)
axis.yaxis.set_tick_params(labelsize=20)

plt.show()