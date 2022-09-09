import torch
from basic_functions import sigmoid
import numpy as np
from time import time
  
def timer_func(func):
    # This function shows the execution time of 
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func
  
v_x = np.random.randn(1, 10)

@timer_func
def time_slow_stable_sigmoid(v_x):
    output = np.zeros(v_x.shape)
    it = np.nditer(v_x, flags=['multi_index'])
    for x in it:
        output[it.multi_index] = sigmoid(x)
    return output

@timer_func
def time_stable_sigmoid(v_x):
    output = sigmoid(v_x)
    return output

@timer_func
def time_pytorch_sigmoid(v_x):
    t_x = torch.tensor(v_x)
    t_output = torch.sigmoid(t_x)
    return t_output.numpy()

@timer_func
def time_slow_pytorch_sigmoid(v_x):
    output = np.zeros(v_x.shape)
    it = np.nditer(v_x, flags=['multi_index'])
    for x in it:
        output[it.multi_index] = torch.sigmoid(torch.tensor(x)).numpy()
    return output

v_y = time_stable_sigmoid(v_x)
v_y1 = time_slow_stable_sigmoid(v_x)
v_y2 = time_pytorch_sigmoid(v_x)
v_y3 = time_slow_pytorch_sigmoid(v_x)
print(np.mean((v_y-v_y2)**2))
print(np.mean((v_y1-v_y3)**2))

