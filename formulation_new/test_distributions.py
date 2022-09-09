from math import floor
import pickle
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np  
import torch
from scipy.interpolate import interp1d
import pdb
from projection_simplex import projection_simplex_sort as proj_simplex

def sample_cdf(data):
    x_ = np.sort(data)
    y_ = (np.arange(len(data))+0.5) / len(data)
    return x_, y_

def learn_single_nonlinearity(x, y, act_function):
    xv = np.reshape(x, (-1, 1))
    v_w = np.arange(1, 7, 0.3)
    v_loc = np.arange(np.min(x), np.max(x), 0.2)
    wm, locm = np.meshgrid(v_w, v_loc)
    m_w = np.reshape(wm, (1,-1))
    m_loc = np.reshape(locm, (1, -1))
    M = m_loc.size
    m_k = m_loc * m_w
    t_H = act_function(torch.tensor(m_w*xv - m_k))
    m_H = t_H.numpy()
    
    b = np.min(y)
    y_range = np.max(y) - b
    lamda = 0.0001
    A = np.transpose(m_H)@m_H+lamda*np.identity(M)
    bb = np.transpose(m_H)@(y-b)
    alpha_init = proj_simplex(np.ones((M)), y_range)
    #print("LS...")
    #alpha_ls = np.linalg.solve(A, bb)
    
    #alpha_sparse = proj_simplex(alpha_ls, y_range)
    #figure, axis = plt.subplots(1)
    #axis.stem(alpha)
    figure, axis = plt.subplots(1)
    axis.plot(np.sort(x), y[np.argsort(x)])
    #yhat = m_H@alpha
    #yhat_ls = m_H@alpha_ls
    yinit = m_H@alpha_init
    #axis.plot(np.sort(x), b+yhat[np.argsort(x)])
    #axis.plot(np.sort(x), b+yhat_ls[np.argsort(x)])
    axis.plot(np.sort(x), b+yinit[np.argsort(x)])
    
    NE = 20
    eta = 0.001
    alpha = alpha_init
    print("SGD...")
    mask = np.ones(M)
    SR = 1./3000 # sparsification rate
    SGOAL = 40  # sparsity goal
    ss = 0 # sparsification status
    for i_epoch in range(NE):
        print("epoch",i_epoch)
        for i_sample in range(x.size):
            v_h = m_H[i_sample, :]
            alpha -= eta*mask*v_h*(v_h@alpha-y[i_sample]+b)
            ss += SR*(sum(mask)- SGOAL)
            idx = np.argpartition(alpha, floor(ss))
            mask[idx[:floor(ss)]] = 0
            alpha = proj_simplex(alpha*mask, y_range)
        print("ss =", ss)
    yhat = m_H@alpha
    axis.plot(np.sort(x), b+yhat[np.argsort(x)])
    figure, axis = plt.subplots(1)
    axis.stem(alpha)
    plt.show()
    pdb.set_trace()


z_data = pickle.load(open("data/synthetic/z_wAs_10_fun_3_sliced.pickle","rb"))
z0 = z_data[0,:]
z_var, y_var = sample_cdf(z0)
cdf_z0 = interp1d(z_var, y_var)
x = cdf_z0(z0)
#pdb.set_trace()
y = norm.ppf(x) # x == norm.cdf(y)

alpha = learn_single_nonlinearity(y, z0, torch.sigmoid)

figure, axis = plt.subplots(1)
axis.set_title("histogram of z0")
plt.hist(z0, bins = 50)

figure, axis = plt.subplots(1)
axis.set_title("histogram of x (uniformly distributed)")
plt.hist(x, bins = 50)

figure, axis = plt.subplots(1)
axis.set_title("histogram of y (Gaussian distributed)")
plt.hist(y, bins = 50)

figure, axis = plt.subplots(2)
axis[0].set_title("f")
axis[1].set_title("f inverse")
axis[0].plot(y[np.argsort(z0)], np.sort(z0))
axis[1].plot(np.sort(z0), y[np.argsort(z0)])

plt.show()