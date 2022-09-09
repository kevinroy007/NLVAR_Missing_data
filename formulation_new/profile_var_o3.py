from operator import ne
import numpy as np
import pickle
import pdb, os, sys

from learn_model import learn_model_init, learn_model_linear

N=10
M=50
P = 2
etal = 0.001
etanl = 0.0001
NE  = 10
NE_l = 10
lamda_n = 0.001 
lamda_l = 0.001
N_init = 1

#np.random.seed(0)


def var():

    file_directory = sys.path[0]
    os.chdir(file_directory)
    #z_data = pickle.load(open("results/A_wAs_10_fun_3_n.txt","rb"))
    #z_data = pickle.load(open("lundin_2000_n.txt","rb"))
    z_data = pickle.load(open("data/synthetic/synthetic_dataset_1.pickle","rb"))
    z_data = z_data
    ############################################################################

    A0 = np.zeros((N, N, P))
    Lcost, Lcost_test, A_l, Lcost_val = \
        learn_model_linear(NE_l, z_data, A0, etal, lamda_n)
    cost,cost_test,A_n,cost_Val = \
        learn_model_init(NE, etanl ,z_data,lamda_n,P, M,N_init, NE_linearinit=10)

    #############################################################################

def main():
    import cProfile
    import pstats

    with cProfile.Profile() as pr:
        var()
    stats = pstats.Stats(pr)
    #stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats(filename='profiling_results3.prof')

if __name__ == "__main__":
    main()