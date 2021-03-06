# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 10:35:47 2019

@author: 13383861
"""

'''
This contains belief p(x_t | z_1:t) update functions, where state is defined by the random variable denoting the location
of a source of something that can be detected, relative to an occupancy grid.
'''
import math
import time
import sys
import typing
from Utils.Vector3r import Vector3r

import numpy as np
import numba

#%%
#version that uses matrix multiplication - for now just use vectors as no point in using more space than 
#necessary    
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

#CUDA version is slower than numpy from initial experiments, forget about it for now
#if numba.cuda.detect():
#    print("Using cuda in the back-end for updates")
#    @numba.vectorize([numba.float64(numba.float64, numba.float64)])
#    def add(m1, m2):
#        return m1 + m2
#    
#    @numba.vectorize([numba.float64(numba.float64, numba.float64)])
#    def mul(m1, m2):
#        return m1 * m2
#    
#    @numba.vectorize([numba.float64(numba.float64, numba.float64)])
#    def div(m1, m2):
#        return m1 / m2
#    
#    @numba.vectorize([numba.float64(numba.float64, numba.float64)])
#    def sub(m1, m2):
#        return m1 - m2
#    
#    #CUDA version
#    @numba.jit(nopython = True)
#    def _get_next_estimated_state_function_neg_reading(matrix_size, a0, b0):
#        #location_matrix = np.zeros((matrix_size, matrix_size))
#        @numba.vectorize([numba.float64(numba.float64, numba.float64)], target = 'cuda')
#        def _get_next_estimated_state(location_index, previous_estimated_state):
#            '''Returns a vector representing the next estimated state given a negative belief reading'''
#            location_matrix = np.zeros((matrix_size, matrix_size))
#            location_matrix[location_index,location_index] = 1
#            #next_estimated_state = previous_estimated_state *(location_matrix * b0 + (np.ones(matrix_size) - location_matrix)*a0)
#            next_estimated_state = mul(previous_estimated_state, sum(mul(location_matrix, b0), mul(sub(np.ones(matrix_size),location_matrix), a0)))
#            #return next_estimated_state/next_estimated_state.trace()
#            return div(next_estimated_state,next_estimated_state.trace())
#        return _get_next_estimated_state
#    
#    def _get_next_estimated_state_function_pos_reading(matrix_size, a1, b1):
#        #location_matrix = np.zeros((matrix_size, matrix_size))
#        @numba.vectorize([numba.float64(numba.float64, numba.float64)], target = 'cuda')
#        def _get_next_estimated_state(location_index, previous_estimated_state):
#            '''Returns a vector representing the next estimated state given a positive belief reading'''
#            location_matrix = np.zeros(matrix_size)
#            location_matrix[location_index] = 1
#            next_estimated_state = previous_estimated_state *(location_matrix * b1 + (np.ones(matrix_size) - location_matrix)*a1)
#            return next_estimated_state/next_estimated_state.trace()
#        return _get_next_estimated_state



        
#@numba.jit(nopython = True)
def _get_next_estimated_state_function_pos_reading(matrix_size, a1, b1):
    @numba.jit(nopython = True, parallel = True)
    def _get_next_estimated_state(location_index, previous_estimated_state):
        '''Returns a vector representing the next estimated state given a positive belief reading'''
        location_matrix = np.zeros(matrix_size)
        location_matrix[location_index] = 1
        next_estimated_state = previous_estimated_state *(location_matrix * b1 + (np.ones(matrix_size) - location_matrix)*a1)
        return next_estimated_state/next_estimated_state.sum()
    return _get_next_estimated_state

#non-cuda version    
#@numba.jit(nopython = True)
def _get_next_estimated_state_function_neg_reading(matrix_size, a0, b0):
    @numba.jit(nopython = True, parallel = True)
    def _get_next_estimated_state(location_index, previous_estimated_state):
        '''Returns a vector representing the next estimated state given a negative belief reading'''
        location_matrix = np.zeros(matrix_size)
        location_matrix[location_index] = 1
        next_estimated_state = previous_estimated_state * (location_matrix * b0 + (np.ones(matrix_size) - location_matrix)*a0)
        return next_estimated_state/next_estimated_state.sum()
    return _get_next_estimated_state


#Allow the user to specify certain locations which have zero probability of detection 
#due to a previous detection having already occurred there



#%%
#@numba.jit(nopython = True)
def gen_next_estimated_state_functions(state_size, a1, b1, a0, b0):    
    '''Generates the functions that estimate the next state given the current state. Since 
    these don't change, they are kept in memory'''
    pos = _get_next_estimated_state_function_pos_reading(state_size, a1, b1)
    neg = _get_next_estimated_state_function_pos_reading(state_size, a0, b0)
    return pos, neg

#@numba.jit(nopython = True)
def get_next_estimated_state(location_index, reading, previous_estimated_state, update_fns):
    '''Returns the next estimated state based on previous state and the most recent sensor reading'''
    if reading == 1:
        update_fn = update_fns[0]
    else:
        update_fn = update_fns[1]
    next_state=update_fn(location_index, previous_estimated_state)
    #return the next estimated state normalised
    return next_state


class BeliefVector:
    '''Belief can be described by a vector. Updates are performed as vector addition and 
    multiplication. By convention the belief vector is in the format:
    (source present at first location, ...., source present at last location, source not present in grid)'''
    def __init__(self, valid_locations: typing.List[Vector3r], initial_state: np.array, fpr, fnr):
        self.fpr = fpr
        self.fnr = fnr
        # a list of valid location objects, whose order corresponds to the estimated state
        self.locations = valid_locations
        #sets up observation and transition matrices
        self._setup_matrices(initial_state)
        #by convention state "0" which represents none of the grid cells containing the source
        #is the last element on the diagonal        
        

    def _setup_matrices(self, initial_state):
        '''Code to setup matrices separated out'''
        #self.estimated_state = np.ones(len(initial_state)) * initial_state
        self.estimated_state = initial_state
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
        assert math.isclose(self.estimated_state.sum(), 1, rel_tol = 0.000001), self.estimated_state.sum()
        
    def _update(self, location_index, reading):
        self.estimated_state = get_next_estimated_state(location_index, reading, self.estimated_state, self.update_fns)
        
    def get_location_index(self,location: Vector3r):
        #this shouldn't need to be used - location indices are ordered by convention
        return self.locations.index(location)
    
    def update(self, location: typing.List[Vector3r], reading):
        '''Given a reading and a location, update the posterior belief in the state given the measurement'''
        location_index = self.get_location_index(location)
#        print("calling _update_calculation with values", location_index, reading)
        self._update(location_index, reading)
    
    def update_from_multiple_measurements(self, locations: typing.List[Vector3r], readings: list):
        '''Updates the posterior p(xt | e1:t) given multiple measurements'''
        assert len(locations) == len(readings)
        for location, reading in zip(locations, readings):
            self.update(location, reading)
        
    def get_estimated_state(self):
        '''Returns the occupancy values for each cell location in list of locations'''
        return self.estimated_state
        
    def get_prob_in_grid(self):
        '''Returns the agent's belief that the source is present in the grid.'''
        return self.estimated_state.sum() - self.estimated_state[-1]
    
class BeliefVectorMultipleSources(BeliefVector):
    '''
    A derived class of BeliefVector which facilitates the recording of the location of previously detected sources, 
    which means that an agent can update it's belief state
    '''
    def __init__(self, valid_locations: typing.List[Vector3r], initial_state: np.array, fpr, fnr):
        super().__init__(valid_locations, initial_state, fpr, fnr)
        
    def set_location_prob_to_zero(self, location):
        #set the location index belief to zero
        self.estimated_state[self.get_location_index(location)] = 0
        #normalize to 1
        self.estimated_state = self.estimated_state/self.estimated_state.sum()
        #from now on, the estimated value of the given state will not change from 0 since
        #updates are performed via multiplication
        #any observations at the modified grid location will have no effect on the update

#%%
class DiscreteBayesFilter:
    
    '''Refer to AI:AMA Page 579 for details. Only valid for a single state variable'''
    
    def __init__(self, transition_model, measurement_models, initial_state):
        '''
        Transition model is the matrix which defines p(Xt | Xt-1). 
        Measurement models is the a dictionary mapping an observation to a 
        measurement model matrix which defines p(Et | Xt). 
        Assuming that user is working with a hidden Markov model.
        '''
        self.transition_model = transition_model
        self.measurement_models = measurement_models
        
    def calculate_forward_pass(self, observation ):
        pass
    
    def calculate_backward_pass(self):
        pass
    
    def get_filter(self):
        pass
    
    def get_smoother(self):
        pass
    
    def get_predictor(self):
        pass
    
    def get_fixed_lag(self):
        pass
    
    def get_most_likely_sequence(self):
        pass
    
    
#%%    
        
#generate some data to test how quick updating is
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
#%%        
    update_fns = gen_next_estimated_state_functions(initial_state.shape[0], a1, b1, a0, b0)
    #%%    
    test_grid = UE4Grid(1,1,Vector3r(0.0), 6, 8)
    prior = {grid_loc:0.008 for grid_loc in test_grid.get_grid_points()}
    initial_state = np.array([prior[grid_point] for grid_point in test_grid.get_grid_points()])
    initial_state = np.append(initial_state,1-initial_state.sum())
    
    valid_locations = test_grid.get_grid_points()
    bel_vec = BeliefVector(valid_locations, initial_state, fpr, fnr)
    bel_vec.get_prob_in_grid()
    #update with a negative reading
    bel_vec.update(Vector3r(2,3), 0)
    bel_vec.assert_well_defined()
    bel_vec.estimated_state.sum()
    #check that updates properly applied
    for belief in bel_vec.estimated_state[:-1]:
        try:
            assert math.isclose(belief,0.008050089445438283, rel_tol = 0.000001), belief
        except AssertionError:
            assert math.isclose(belief,0.0017889087656529517, rel_tol = 0.000001), belief
    
    #%%
    #check that can successfully set the belief that a source in certain grid locations are zero
    bel_vec_mul_sources = BeliefVectorMultipleSources(valid_locations, initial_state, fpr, fnr)
    bel_vec_mul_sources.update(Vector3r(2,3), 0)
    bel_vec_mul_sources.assert_well_defined()
    
    bel_vec_mul_sources.set_location_prob_to_zero(Vector3r(5,0))
    assert bel_vec_mul_sources.estimated_state[5] == 0
    bel_vec_mul_sources.update(Vector3r(4,3), 1)
    assert bel_vec_mul_sources.estimated_state[5] == 0
    
    
    
    
    
    
    #%%
    #Plot some timings
    test_grid_sizes = np.array([i**2 for i in range(100,2000,100)])
    #print("size of test_grids: ", sys.getsizeof(test_grids))
    timings = np.zeros(len(test_grid_sizes))
    sizes = np.zeros(len(test_grid_sizes))
    _plot_update_stats(timings, fpr, fnr, test_grid_sizes)
    
    #%%Ensure beliefs are updated as expected
