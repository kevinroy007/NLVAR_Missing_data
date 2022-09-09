import sys
from sqlalchemy import false
sys.path.append(sys.path[0]+'/..')

import torch
import matplotlib.pyplot as plt
import pdb
import pickle
import numpy as np
import random
from torch import nn
from basic_functions import f_param_torch
from torch.nn import Parameter
from basic_functions import f_param

import os
os.system("clear")

def learn_initial_nonlinearity(z_lower, z_upper, M,i):
    Nsamples = 1000
    # generate samples
    a = (2*z_lower-z_upper)
    b = (2*z_upper - z_lower)
    x_range = torch.zeros(Nsamples)
    y_range = torch.zeros(Nsamples)

    for t in range(Nsamples):
        x_range[t] = random.uniform(a,b)


    #x_range = torch.arange(2*z_lower-z_upper, 2*z_upper - z_lower, (b-a)/(Nsamples))
    #pdb.set_trace()
    try:
        assert x_range.numel() == Nsamples
    except:
        pdb.set_trace()

    

    for t in range(Nsamples):
        if (z_lower - (z_upper - z_lower)) <= x_range[t] <= z_lower:
            y_range[t] = z_lower
        elif z_upper <= x_range[t] <= (z_upper + (z_upper - z_lower)): 
            y_range[t] = z_upper
        else:
            y_range[t] = x_range[t]
    

    return x_range,y_range
    #plt.plot(x_range,y_range)
    #plt.show()


def sample_generation(z_data,M):
    
    N,T = z_data.shape
    M = 10
    x_range = torch.zeros(N,T)
    y_range = torch.zeros(N,T)

    for i in range(N):
        z_upper = np.max(z_data[i,:])
        z_lower = np.min(z_data[i,:])
        x_range[i,:],y_range[i,:] = learn_initial_nonlinearity(z_lower,z_upper,M,i)
    return x_range,y_range


input_data_filename = "../data/synthetic/synthetic_dataset_1.pickle"

#input_data_filename = "../data/real/lundin_normalised.pickle"

z_data = pickle.load(open(input_data_filename,"rb"))
N,T = z_data.shape
M = 10
x_range,y_range = sample_generation(z_data,M)



alpha = torch.zeros((N,M),requires_grad=True)
b = torch.zeros((N),requires_grad=True)
k = torch.zeros((N,M),requires_grad=True)
w = torch.zeros((N,M),requires_grad=True)

model =[Parameter(alpha),Parameter(w),Parameter(k),Parameter(b)]
lr = 0.1
criterion = nn.MSELoss()
optimzer =torch.optim.SGD(model,lr = lr,momentum=0.9)

dict  ={"alpha":alpha,"w":w,"k":k,"b":b}


learn_para = True

if learn_para:
    for epoch in range(1000):
        optimzer.zero_grad()
        y_pred = torch.zeros(N,T)
        for i in range(N):
            for t in range(T):
                y_pred[i,t] = f_param_torch(x_range[i,t],i,model[0],model[1],model[2],model[3])
        
        loss = criterion(y_pred,y_range)
        #pdb.set_trace()
        loss.backward()
        optimzer.step()
        print("epoch",epoch,"loss",loss)
# plt.plot(x_range[1,:],y_range[1,:], "bo")
# plt.show()
    alpha = model[0].detach().numpy()
    w = model[1].detach().numpy()
    k = model[2].detach().numpy()
    b = model[3].detach().numpy()

    dict  ={"alpha":alpha,"w":w,"k":k,"b":b}
    #
    # pickle.dump(dict,open("linear_para_dict.pickle","wb"))
    #pickle.dump(dict,open("linear_para_dict_lundin.pickle","wb"))

check_linear_func = True

if check_linear_func:

    x = np.arange(-10,10,0.01)
    y = np.zeros(len(x))

    for t in range(len(x)):
        y[t] = f_param(x[t],1, alpha, w, k, b)
    
    pdb.set_trace()
    plt.plot(x,y,"ro")
    plt.show()