# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 10:35:47 2019

@author: 13383861
"""

import math
import time
import sys

import numpy as np
import numba

#%%
    
##@numba.jit(nopython = True)
#def _get_next_estimated_state_function_pos_reading(matrix_size, a1, b1):
#    #location_matrix = np.zeros((matrix_size, matrix_size))
#    @numba.jit(nopython = True, parallel = True)
#    def _get_next_estimated_state(location_index, previous_estimated_state):
#        location_matrix = np.zeros((matrix_size, matrix_size))
#        location_matrix[location_index,location_index] = 1
#        return previous_estimated_state *(location_matrix * b1 + (np.identity(matrix_size) - location_matrix)*a1)
#    return _get_next_estimated_state
#    
##@numba.jit(nopython = True)
#def _get_next_estimated_state_function_neg_reading(matrix_size, a0, b0):
#    #location_matrix = np.zeros((matrix_size, matrix_size))
#    @numba.jit(nopython = True, parallel = True)
#    def _get_next_estimated_state(location_index, previous_estimated_state):
#        location_matrix = np.zeros((matrix_size, matrix_size))
#        location_matrix[location_index,location_index] = 1
#        next_estimated_state = previous_estimated_state *(location_matrix * b0 + (np.identity(matrix_size) - location_matrix)*a0)
#        return next_estimated_state/next_estimated_state.trace()
#    return _get_next_estimated_state
#
##@numba.jit(nopython = True)
#def gen_next_estimated_state_functions(state_size, a1, b1, a0, b0):    
#    pos = _get_next_estimated_state_function_pos_reading(state_size, a1, b1)
#    neg = _get_next_estimated_state_function_pos_reading(state_size, a0, b0)
#    return pos, neg
#
##@numba.jit(nopython = True)
#def get_next_estimated_state(location_index, reading, previous_estimated_state, update_fns):
#    if reading == 1:
#        update_fn = update_fns[0]
#    else:
#        update_fn = update_fns[1]
#    next_state=update_fn(location_index, previous_estimated_state)
#    return next_state/next_state.trace()

#%%
#@numba.jit(nopython = True)
def _get_next_estimated_state_function_pos_reading(matrix_size, a1, b1):
    #location_matrix = np.zeros((matrix_size, matrix_size))
    @numba.jit(nopython = True, parallel = True)
    def _get_next_estimated_state(location_index, previous_estimated_state):
        location_matrix = np.zeros(matrix_size)
        location_matrix[location_index] = 1
        return previous_estimated_state *(location_matrix * b1 + (np.ones(matrix_size) - location_matrix)*a1)
    return _get_next_estimated_state
    
#@numba.jit(nopython = True)
def _get_next_estimated_state_function_neg_reading(matrix_size, a0, b0):
    #location_matrix = np.zeros((matrix_size, matrix_size))
    @numba.jit(nopython = True, parallel = True)
    def _get_next_estimated_state(location_index, previous_estimated_state):
        location_matrix = np.zeros((matrix_size, matrix_size))
        location_matrix[location_index,location_index] = 1
        next_estimated_state = previous_estimated_state *(location_matrix * b0 + (np.ones(matrix_size) - location_matrix)*a0)
        return next_estimated_state/next_estimated_state.trace()
    return _get_next_estimated_state

#@numba.jit(nopython = True)
def gen_next_estimated_state_functions(state_size, a1, b1, a0, b0):    
    pos = _get_next_estimated_state_function_pos_reading(state_size, a1, b1)
    neg = _get_next_estimated_state_function_pos_reading(state_size, a0, b0)
    return pos, neg

#@numba.jit(nopython = True)
def get_next_estimated_state(location_index, reading, previous_estimated_state, update_fns):
    if reading == 1:
        update_fn = update_fns[0]
    else:
        update_fn = update_fns[1]
    next_state=update_fn(location_index, previous_estimated_state)
    return next_state/next_state.sum()


class BeliefVector:
    '''Belief can be described by a vector. Updates are performed as vector addition and 
    multiplication'''
    def __init__(self, valid_locations, initial_state, fpr, fnr):
        self.fpr = fpr
        self.fnr = fnr
        # a list of valid locations, whose order corresponds to the estimated state
        self.locations = valid_locations
        self._setup_matrices(initial_state)
        #by convention state "0" which represents none of the grid cells containing the source
        #is the last element on the diagonal        

    def _setup_matrices(self, initial_state):
        '''Code to setup matrices separated out'''
        self.estimated_state = np.ones(len(initial_state)) * initial_state
        self.identity = np.ones(len(initial_state))
        self.fpr_matrix = self.fpr * np.ones(len(initial_state))
        self.fnr_matrix = self.fnr * np.ones(len(initial_state))
        
        self.b0 = self.fnr_matrix
        self.a0 = self.identity - self.fpr_matrix
        
        self.b1 = self.identity - self.fnr_matrix
        self.a1 = self.fpr_matrix
        
        self.matrix_size = len(initial_state)
        self.update_fns = gen_next_estimated_state_functions(self.matrix_size, self.a1, self.b1, self.a0, self.b0)
        self.assert_well_defined()
    
    def assert_well_defined(self):
        '''checks that trace of estimated state is 1'''
        assert math.isclose(self.estimated_state.trace(), 1, rel_tol = 0.000001), self.estimated_state.trace()
        
    def _update(self, location_index, reading):
        self.estimated_state = get_next_estimated_state(location_index, reading, self.estimated_state, self.update_fns)
        
    def get_location_index(self,location):
        return self.locations.index(location)
    
    def update(self, location, reading):
        location_index = self.get_location_index(location)
#        print("calling _update_calculation with values", location_index, reading)
        self._update(location_index, reading)      
        
    def get_estimated_state(self):
        return self.estimated_state
        
    def get_prob_in_grid(self):
        return self.estimated_state.trace() - self.estimated_state[self.estimated_state.shape[0]-1]

#%%    
def _gen_data(timings, fpr, fnr, test_grid_sizes):
    for test_grid_index, test_grid_size in enumerate(test_grid_sizes):
        print(test_grid_index)
        initial_state = np.array([0.000000008 for i in range(test_grid_size)])
        initial_state = np.append(initial_state, 1-initial_state.sum())
        #* np.ones(len(initial_state))
        fpr_matrix = fpr * np.ones(len(initial_state))
        fnr_matrix = fnr * np.ones(len(initial_state))
        identity = np.ones(len(initial_state))
        b0 = fnr_matrix
        a0 = identity - fpr_matrix
        b1 = identity - fnr_matrix
        a1 = fpr_matrix
        
        update_fns = gen_next_estimated_state_functions(initial_state.shape[0], a1, b1, a0, b0)
    
    
    #        @numba.jit(nopython = True)
        #def gen_timings():
        t1 = time.time()
        for i in range(5):
            next_state = get_next_estimated_state(8, 0, initial_state, update_fns)
            next_state = get_next_estimated_state(10, 1, next_state, update_fns)
            next_state = get_next_estimated_state(12, 0, next_state, update_fns)
            next_state = get_next_estimated_state(15, 0, next_state, update_fns)
            next_state = get_next_estimated_state(20, 0, next_state, update_fns)
            next_state = get_next_estimated_state(21, 0, next_state, update_fns)
            next_state = get_next_estimated_state(22, 1, next_state, update_fns)
        #    return 0
    #        gen_timings()
        t2 = time.time()
        timings[test_grid_index] = (t2 - t1)/35
            #return next_state
        next_state = get_next_estimated_state(15, 0, initial_state, update_fns)
        sizes[test_grid_index] = (sys.getsizeof(next_state) + sys.getsizeof(update_fns) + sys.getsizeof(b0) + sys.getsizeof(b1) +
             sys.getsizeof(a0) + sys.getsizeof(a1) + sys.getsizeof(fpr_matrix) + sys.getsizeof(fnr_matrix))
    return timings, sizes
#%%
    
def _plot_update_stats(timings, fpr, fnr, sizes):
    timings, sizes = _gen_data(timings, fpr, fnr, sizes)
    print(timings)
    print(sizes)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot([i**2 for i in range(100,2200, 100)], timings)
    plt.title("Average time taken to update vs. grid size")
    #plt.xlim(0,2200)
    
    plt.figure()
    plt.plot([i**2 for i in range(100,2200, 100)], sizes)
    plt.title("Average size in bytes taken to update vs. grid size")
    #plt.xlim(0, 2200)  
#%%
if __name__ == '__main__':
    #%%
    from Utils.UE4Grid import UE4Grid
    from Utils.Vector3r import Vector3r
    fpr = 0.1
    fnr = 0.2
    test_grid = UE4Grid(1,1,Vector3r(0.0), 20, 20)
    
    initial_state = [0.008 for i in range(len(test_grid.get_grid_points()))]
    initial_state.append(1-sum(initial_state))
    initial_state = initial_state * np.identity(len(initial_state))
    fpr_matrix = fpr * np.identity(len(initial_state))
    fnr_matrix = fnr * np.identity(len(initial_state))
    identity = np.identity(len(initial_state))    
    b0 = fnr_matrix
    a0 = identity - fpr_matrix
    b1 = identity - fnr_matrix
    a1 = fpr_matrix
        
    update_fns = gen_next_estimated_state_functions(initial_state.shape[0], a1, b1, a0, b0)
    #%%
    #test_grids = [UE4Grid(1,1,Vector3r(0.0), i, i) for i in range(100,2200, 100)]
    test_grid_sizes = np.array([i**2 for i in range(100,2200, 100)])
    #print("size of test_grids: ", sys.getsizeof(test_grids))
    timings = np.zeros(len(test_grid_sizes))
    sizes = np.zeros(len(test_grid_sizes))
    
    _plot_update_stats(timings, fpr, fnr, test_grid_sizes)
    
    
    
    
    
    
    
    
    
    
    
    