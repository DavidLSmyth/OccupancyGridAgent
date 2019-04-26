# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 22:23:08 2019

@author: 13383861
"""

import numpy as np
from timeit import default_timer as timer
import numba


@numba.vectorize(['float32(float32, float32)'], target='cuda')
def pow(a, b):
    return a ** b

def main():
    vec_size = 100000000

    a = b = np.array(np.random.sample(vec_size), dtype=np.float32)
    c = np.zeros(vec_size, dtype=np.float32)

    start = timer()
    c = pow(a, b)
    duration = timer() - start

    print(duration)


main()


@numba.cuda.jit
def increment_by_one(an_array):
    pos = cuda.grid(1)
    if pos < an_array.size:
        an_array[pos] += 1
        
        
@numba.cuda.jit
def increment_a_2D_array(an_array):
    x, y = numba.cuda.grid(2)
    if x < an_array.shape[0] and y < an_array.shape[1]:
       an_array[x, y] += 1
     
@numba.cuda.jit
def sum_1DArrays(array1, array2, return_array):
    x = numba.cuda.grid(1)
    if x < array1.shape[0]:
       return_array[x] = array1[x] + array2[x]
    
    
import math
threadsperblock = (32, 32)
#an_array = np.array(([1,2,3,4,5], [2,3,4,5,1]))
array1 = np.arange(0, 1000000, 2)
array2 = np.arange(0, 1000000, 2)
return_array = np.zeros(array1.shape[0])

blockspergrid_x = math.ceil(array1.shape[0] / threadsperblock[0])
blockspergrid = (blockspergrid_x, )
import time
t1 = time.time()
sum_1DArrays[blockspergrid, threadsperblock](array1, array2, return_array)
sum_1DArrays[blockspergrid, threadsperblock](return_array, array2, return_array)
t2 = time.time()
print(t2 - t1)
t3 = time.time()
res = array2 + array1
res = res + array2
t4 = time.time()
print(t4 - t3)
#for some reason CUDA version is slower than CPU


blockspergrid_x = math.ceil(an_array.shape[0] / threadsperblock[0])
blockspergrid_y = math.ceil(an_array.shape[1] / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)
increment_a_2D_array[blockspergrid, threadsperblock](an_array)
