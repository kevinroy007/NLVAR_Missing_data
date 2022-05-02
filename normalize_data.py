import numpy as np
import pickle
import pdb

import matplotlib.pyplot as plt

# z_data = pickle.load(open("results/A_wAs_10_fun.txt","rb"))

# N,T = z_data.shape



z_data = pickle.load(open("results/A_wAs_10_fun_3_n.txt","rb"))
#z_data = pickle.load(open("results/A_wAs_10_fun_2000.txt","rb"))
#z_data = pickle.load(open("lundin_2000_n.txt","rb"))

pdb.set_trace()
N,T = z_data.shape
z_data_1 = np.random.rand(N,T)

for i in range(N):
        z_data_1[i,:] = (z_data[i,:] - np.mean(z_data[i,:]))/np.std(z_data[i,:])
for i in range(N):
        z_data_1[i,:] = z_data[i,:]/np.max(z_data[i,:])


plt.hist(z_data_1[2,:], bins = 100)

#pickle.dump(z_data_1,open("lundin_2000_n_1.txt","wb"))
#pickle.dump(z_data_1,open("results/A_wAs_10_fun_2000_n_1.txt","wb"))

pdb.set_trace()

plt.show()

