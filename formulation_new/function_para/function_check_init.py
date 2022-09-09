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


check_linear_func = True
dict = pickle.load(open("linear_para_dict.pickle","rb"))

alpha = dict["alpha"]
w = dict["w"]
k = dict["k"]
b = dict["b"]



if check_linear_func:

    x = np.arange(-10,10,0.01)
    y = np.zeros(len(x))

    for t in range(len(x)):
        y[t] = f_param(x[t],1, alpha, w, k, b)
    
    pdb.set_trace()
    plt.plot(x,y,"ro")
    plt.show()