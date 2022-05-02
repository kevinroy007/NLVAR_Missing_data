import numpy as np
from matplotlib import rc,rcParams
import matplotlib.pyplot as plt
import pickle

def f(x,i): 


        a = b[i]
        for m in range(M):
            a = a + alpha[i][m] * sigmoid(w[i][m]*x-k[i][m])
  
        return a
def sigmoid(l):
        
        return  1/(1+np.exp(-l)) 

N = 10
P =2
M =10

alpha = np.random.rand(N,M)
w = np.random.rand(N,M)*20
k = np.random.rand(N,M)*10
b = np.random.rand(N)
A = np.random.rand(N,N,P)

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

# pickle.dump(alpha,open("function_para/alpha.txt","wb"))
# pickle.dump(w,open("function_para/w.txt","wb"))
# pickle.dump(k,open("function_para/k.txt","wb"))
# pickle.dump(b,open("function_para/b.txt","wb"))
# pickle.dump(A,open("function_para/A.txt","wb"))

    
plt.show()

