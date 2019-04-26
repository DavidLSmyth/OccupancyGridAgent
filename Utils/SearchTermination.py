# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 19:08:57 2019

@author: 13383861
"""

#This file contains the code that determines when to terminate the search. Chung & Burdick suggest in "A Decision-Making Framework for Control Strategies
#in Probabilistic Search" to use the Sequential Probability Ratio Test. For the given framework, it can be shown that the SPRT is optimal in the sense 
#that it minimizes the average sample size before a decision is
#made among all sequential tests which do not have larger error probabilities than the SPRT.

#The idea behind the sequential testing is that we collect observations one at a time; when observation Xi = xi has been made, we choose between the following options:
#• Accept the null hypothesis and stop observation;
#• Reject the null hypothesis and stop observation;
#• Defer decision until we have collected another piece of information as Xi+1.

#The challenge is now to find out when to choose which of the above options. We would want to control the two types of error
#alpha = P{Deciding for HA when H0 is true} and beta = P{Deciding for H0 when HA is true}.


import math
import typing
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from Utils.BeliefMap import (SingleSourceBinaryBeliefMap, BeliefMapComponent, get_lower_and_upper_confidence_given_obs, 
    calc_single_source_posterior_given_sensor_sensitivity, calculate_binary_sensor_probability)
from Utils.ObservationSetManager import ObservationSetManager
import sys
sys.path.append('..')

from Utils.UE4Grid import UE4Grid
#from AirSimInterface.types import Vector3r
from Utils.Vector3r import Vector3r
from Utils.AgentObservation import AgentObservation#, BinaryAgentObservation
from Utils.Prior import generate_gaussian_prior, generate_uniform_prior

#%%
#It can also be shown that the boundaries A and B can be calculated as with very good approximation as
#A = log(beta/(1-alpha))      B = log((1-beta)/alpha)
#alpha = P{Deciding for HA when H0 is true} and beta = P{Deciding for H0 when HA is true}.
def _get_upper_bound(alpha, beta):
    return math.log((1-beta)/alpha)

def _get_lower_bound(alpha, beta):
    return math.log(beta/(1-alpha))

def should_terminate_search(s_i):
    return True if (should_accept_h_0(s_i) or should_accept_h_1(s_i)) else False

def log_likelihood_ratio(s_i, bel_map):
    '''
    Here I want to determine the likelihood ratio between the alternative hypothesis and the null hypothesis. Assuming that the null hypothesis is 
    that the source is not present, calculated as L(theta_0 | x) / L(theta_1 | x) = P(x | H = 0) / P(x | H = 1) using Chung & Burdick notation
    '''
    if not isinstance(bel_map, SingleSourceBinaryBeliefMap):
        raise NotImplementedError("Calculating the log likelihood ratio is only implemented for the single source framework")
    else:
        
        raise NotImplementedError("Haven't found a clean closed formula for this yet")

def get_next_s_i(s_i):
    next_s_i = s_i + log_likelihood_ratio(s_i)
    return next_s_i

def should_accept_h_0(s_i, alpha, beta):
    return get_next_s_i(s_i) < _get_lower_bound(alpha, beta)

def should_accept_h_1(s_i, alpha, beta):
    return get_next_s_i(s_i) < _get_upper_bound(alpha, beta)

class SingleSourceSearchTermination(ABC):
    '''Superclass of other classes that determine when to terminate search'''
    def __init__(self):
        pass
    
    @abstractmethod
    def should_end_search(self, belief_map) -> bool:
        '''Given a belief map, returns true if the search should be terminated or false if the search should continue.'''
        if not isinstance(belief_map, SingleSourceBinaryBeliefMap):
            raise NotImplementedError("Calculating the log likelihood ratio is only implemented for the single source framework")
        else:
            raise NotImplementedError("This is a base class - use a subclass")
    
class UpperLowerBoundTotalBeliefSearchTermination(SingleSourceSearchTermination):
    '''A class which terminates the search if the total belief exceeds a user-specified upper bound or a lower bound'''
    def __init__(self, lower_belief_bound, upper_belief_bound):
        assert lower_belief_bound > 0
        assert upper_belief_bound < 1
        assert lower_belief_bound <= upper_belief_bound
        self.upper_belief_bound = upper_belief_bound
        self.lower_belief_bound = lower_belief_bound
        
    def should_end_search(self, belief_map) -> bool:
        '''Given a belief map, returns true if the search should be terminated or false if the search should continue.'''
        if not isinstance(belief_map, SingleSourceBinaryBeliefMap):
            raise NotImplementedError("Total belief search termination is only valid for single-source belief maps")
        
        return self._accept_source_in_grid(belief_map) or self._accept_source_not_in_grid(belief_map)
        
    def _accept_source_in_grid(self, belief_map: SingleSourceBinaryBeliefMap) -> bool:
        return belief_map.get_probability_source_in_grid() > self.upper_belief_bound
    
    def _accept_source_not_in_grid(self, belief_map: SingleSourceBinaryBeliefMap) -> bool:
        return belief_map.get_probability_source_in_grid() < self.lower_belief_bound    
    
class ConfidenceIntervalSearchTermination(UpperLowerBoundTotalBeliefSearchTermination):
    '''
    A class which terminates the search if the total belief exceeds a user-specified upper bound
    and the confidence interval around the grid location at which the belief is maximum is within a certain threshold
    or the total belief falls below a user-specified lower bound
    '''
    def __init__(self, upper_belief_bound, lower_belief_bound, confidence_width):
        super().__init__(upper_belief_bound, lower_belief_bound)
        self.confidence_width = confidence_width
    
    
    def should_end_search(self, belief_map: SingleSourceBinaryBeliefMap, observation_set_manager: ObservationSetManager) -> bool:
        '''Given a belief map, returns true if the search should be terminated or false if the search should continue.'''
        if self._accept_source_in_grid(belief_map):
            greatest_likelihood_component = belief_map.get_most_likely_component()
            greatest_likelihood_component_observations = observation_set_manager.get_all_observations_at_grid_loc(greatest_likelihood_component.grid_loc)
            confidence_bounds = get_lower_and_upper_confidence_given_obs(greatest_likelihood_component_observations)
            #check if the confidence interval is narrow enough
            return confidence_bounds[1] - confidence_bounds[0] < self.confidence_width
        else:
            return False
        
        
class IndividualGridCellSearchTermination(UpperLowerBoundTotalBeliefSearchTermination):
    '''
    A class which terminates the search if the total belief exceeds a user-specified upper bound
    and the difference between the highest individual grid cell belief and the second highest grid cell belief
    is sufficiently large.
    '''
    def __init__(self, upper_belief_bound, lower_belief_bound, min_belief_difference):
        super().__init__(upper_belief_bound, lower_belief_bound)
        #the difference in belief between the highest value and the second highest value
        self.min_belief_difference = min_belief_difference
    
    def should_end_search(self, belief_map: SingleSourceBinaryBeliefMap, observation_set_manager: ObservationSetManager) -> bool:
        '''
        Given a belief map, returns true if the search should be terminated or false if the search should continue.
        '''
        #first check that overall probability of source being in grid exceeds upper bound
        if self._accept_source_in_grid(belief_map):
            #check if the confidence interval is narrow enough
            greatest_likelihood_component = belief_map.get_most_likely_component()
            #get the second most likely component
            second_greatest_likelihood_component = belief_map.get_ith_most_likely_component(2)
            print(greatest_likelihood_component.likelihood - second_greatest_likelihood_component.likelihood)
            return (greatest_likelihood_component.likelihood - second_greatest_likelihood_component.likelihood) > self.min_belief_difference
        else:
            return False
        

class SequentialProbRatioTest(SingleSourceSearchTermination):
    '''
    A class which terminates the search according to Wald's Sequential Likelihood Ratio Test.
    https://en.wikipedia.org/wiki/Sequential_probability_ratio_test.
    The calculation in this case is as follows: 
    p(E1:t|H) = p(H|E1:t)*p(E1:t)/p(H)
    then p(E1:t|H=1)/p(E1:t|H=0) = p(H=1|E1:t)*p(E1:t)/p(H=1) / p(H=0|E1:t)*p(E1:t)/p(H=0)
    
    = sum over all grid locations i ( p(source at location i | E1:t) * 1/prior(source present) / p(source not present | E1:t) * 1/prior(source not presen))
    taking log = log(belief source present at time t) + log(prior(source not present)) - log(belief source not present at time t) - log(prior(source present))
    
    ----------------------------------------------------------------------------------------------------------------------
    
    if both alpha = probability_of_falsely_rejecting_source_is_present_given_source_is_present and beta = probability_of_falsely_accepting_source_is_present_given_source_is_not_present
    are low, then a = beta / 1-alpha will be 'small' (close to 0) and 1-alpha / beta will be 'big' (tending to inf).
    Then the log(a) will be << 0 and log(b) will be >> 0, which means that belief that the source is present will need to be high (and correspondingly log(p(source present given observed data)) will be high)
    in order for the termination criteria to be met. In the opposite case where alpha and beta are both 'big' (close to 1), then the decision region will be small
    '''
    def __init__(self, prior_belief_present, probability_of_falsely_rejecting_source_is_present_given_source_is_present:"p(type 1 error)", probability_of_falsely_accepting_source_is_present_given_source_is_not_present:"p(type 2 error)"):
        self.lower_bound = math.log(probability_of_falsely_accepting_source_is_present_given_source_is_not_present/(1-probability_of_falsely_rejecting_source_is_present_given_source_is_present))
        self.upper_bound = math.log((1-probability_of_falsely_accepting_source_is_present_given_source_is_not_present)/probability_of_falsely_rejecting_source_is_present_given_source_is_present)
        #self.prior = prior
        #self.prior_belief_source_present = prior[:-1].sum()
        #self.prior_belief_source_not_present = 1 - self.prior_belief_source_present 
        self.prior_log_difference = math.log(1-prior_belief_present) - math.log(prior_belief_present) 
        
    def should_end_search(self, current_belief_source_present: 'The belief at the current time that the source is present, given all evidence up to the current time'):
        return self.accept_source_in_grid() or self.accept_source_not_in_grid()

    def get_log_likelihood_ratio(self, current_belief_source_present: 'The belief at the current time that the source is present, given all evidence up to the current time'):
        '''
        Returns the log likelihood ratio log( p(E1:t | source is present) / p(E1:t | source is not present) )
        '''
        return math.log(current_belief_source_present) - math.log(1-current_belief_source_present) + self.prior_log_difference
    
    def get_critical_region(self):
        '''
        Returns the critical region in which there is not enough evidence to accept or reject the null hypothesis.
        This is given by a lower bound and an upper bound
        '''
        return self.lower_bound, self.upper_bound
    
    def plot_critical_region(self):
        '''
        Plots the cutoffs with varying belief in source presence
        '''
        plt.clf()
        point_range = list(range(1,100))
        plt.plot([i/point_range[-1] for i in point_range], [self.get_log_likelihood_ratio(i/100) for i in point_range], label = 'Varying belief whether source present')
        plt.plot([i/point_range[-1] for i in point_range], [self.upper_bound for i in point_range], label = 'Values that exceed accept source not present')
        plt.plot([i/point_range[-1] for i in point_range], [self.lower_bound for i in point_range], label = 'Values that are lower accept source present')
        points_to_fill_between = list(filter(lambda point: self.lower_bound <= self.get_log_likelihood_ratio(point/100), point_range))
        
        points_to_fill_between = list(filter(lambda point: self.upper_bound >= self.get_log_likelihood_ratio(point/100), points_to_fill_between))
        
        plt.fill_between([point/point_range[-1] for point in points_to_fill_between], [self.lower_bound for _ in points_to_fill_between], [self.upper_bound for _ in points_to_fill_between], alpha = 0.3)
        plt.xlabel("Agent belief source is present")
        plt.ylabel("Log-likelihood ratio of agent belief given source is present / agent belief given source is not present")
        plt.legend()
        
    def plot_lower_and_upper_decision_boundary_fixed_type1_error(self, type1_error_rate):
        type2_error_rates = [i/100 for i in range(1,100)]
        lower_bounds = [math.log(type2_error_rate/(1-type1_error_rate)) for type2_error_rate in type2_error_rates]
        upper_bounds = [math.log((1-type2_error_rate)/type1_error_rate) for type2_error_rate in type2_error_rates]
        plt.clf()
        plt.plot([i for i in type2_error_rates], [lower_bound for lower_bound in lower_bounds], label = 'Lower bound varying with type2 error rate')
        plt.plot([i for i in type2_error_rates], [upper_bound for upper_bound in upper_bounds], label = 'Upper bound varying with type2 error rate')
        plt.title("Varying lower bound with type 1 error rate fixed at {}".format(type1_error_rate))
        plt.xlabel("Type2 error rate")
        plt.ylabel("Log-likelihood upper and lower decision thresholds")
        plt.legend()

        
    def plot_lower_and_upper_decision_boundary_fixed_type2_error(self, type2_error_rate):
        type1_error_rates = [i/100 for i in range(1,100)]
        lower_bounds = [math.log(type2_error_rate/(1-type1_error_rate)) for type1_error_rate in type1_error_rates]
        upper_bounds = [math.log((1-type2_error_rate)/type1_error_rate) for type1_error_rate in type1_error_rates]
        plt.clf()
        plt.plot([i for i in type1_error_rates], [lower_bound for lower_bound in lower_bounds], label = 'Lower bound varying with type1 error rate')
        plt.plot([i for i in type1_error_rates], [upper_bound for upper_bound in upper_bounds], label = 'Upper bound varying with type1 error rate')
        plt.title("Varying lower bound with type 2 error rate fixed at {}".format(type2_error_rate))
        plt.xlabel("Type1 error rate")
        plt.ylabel("Log-likelihood upper and lower decision thresholds")
        plt.legend()

    
    def plot_lower_decision_boundary_fixed_type2_error(self, type2_error_rate):
        pass
    
    def accept_source_in_grid(self, current_belief_source_present: 'The belief at the current time that the source is present, given all evidence up to the current time'):
        return self.get_log_likelihood_ratio(current_belief_source_present) <= self.lower_bound
        #return self.log_likelihood_calculator.get_current_log_likelihood() <= self.lower_bound
    
    def accept_source_not_in_grid(self, current_belief_source_present: 'The belief at the current time that the source is present, given all evidence up to the current time'):
        return self.get_log_likelihood_ratio(current_belief_source_present) >= self.upper_bound
        
    
#%%
#The log likelihood can be calculated as 
#log(sum over all locations from a=1 to |A| (product from time k = 1 to t(p(evidence at time k | source is at location a)))) - log(product from time k = 1 to t (p(evidence at time k | source not present)))

#Error here that ratio is not valid
#class LogLikelihoodCalculator:
#    '''
#    A class that helps maintain useful variables for calculating and updating log liklihood.
#    '''
#    
#    def __init__(self, fpr:float, fnr:float, prior: typing.List[float]):
#        #for each state, record the product of the probability of evidence given the state
#        #initialise as an array of ones, so that multiplication can be performed for subsequent observations
#        #for now assuming by convention that states are ordered 1, ..., n, n+1 where the n+1th state corresponds to no
#        #source present in grid
#        
#        #the likelihood product contains at index i:
#        #for all indices but last
#        # product p(observations as far as time t , in state i | source is present)
#        #for last index:
#        # product p(observations as far as time t , in state i | source is not present)
#        #where state = {source at grid loc 1, ..., source at grid loc n, source not present}
#        self.likelihood_product = np.array(prior, dtype = np.float64)
#        self.fpr = fpr
#        self.tnr = 1 - self.fpr
#        self.fnr = fnr
#        self.tpr = 1 - self.fnr
#        self.pos_observation_update_vector = np.array([self.fpr for i in range(len(self.likelihood_product))])
#        self.neg_observation_update_vector = np.array([self.tnr for i in range(len(self.likelihood_product))])
#        self.no_updates = 0
#        
#    def update(self, observation_value:int, observation_location_index: int):
#        '''
#        Updates likelihood vector based on the most recent observation.
#        '''
#        if observation_location_index >= len(self.likelihood_product):
#            raise Exception("Tried to update with an invalid observation. Observations must be gathered in range {}. Index {} is outside this range".format(len(self.likelihood_product) - 1, observation_location_index))
#        if observation_value == 1:
#            update_vector = self.pos_observation_update_vector.copy()
#            update_vector[observation_location_index] = self.tpr
#        else:
#            update_vector = self.neg_observation_update_vector.copy()
#            update_vector[observation_location_index] = self.fnr
#        #update likelihood once the update_vector has been set
#        print(self.likelihood_product)
#        print(update_vector)
#        self.likelihood_product*=update_vector
#        self.no_updates += 1
#    
#    def get_current_log_likelihood(self):
#        if self.no_updates == 0:
#            return 0
#        else:
#            return math.log(self.likelihood_product[:-1].sum()) - math.log(self.likelihood_product[-1])
    
    
class LogLikelihoodCalculator:
    '''
    A class that helps maintain useful variables for calculating and updating log liklihood.
    '''
    
    def __init__(self, fpr:float, fnr:float, prior: typing.List[float]):
        #for each state, record the product of the probability of evidence given the state
        #initialise as an array of ones, so that multiplication can be performed for subsequent observations
        #for now assuming by convention that states are ordered 1, ..., n, n+1 where the n+1th state corresponds to no
        #source present in grid
        self.fpr = fpr
        self.fnr = fnr
        

    
#%%
def calculate_data_probability(sequence, locations, possible_source_location, fpr, fnr):
    '''sequence is a list of 0, 1. Locations is a sequence of Vector3r, agent_location
    is a vector3r'''
    prior = 1
    for location, reading in zip(locations, sequence):
        prior *= calculate_binary_sensor_probability(reading, location, possible_source_location, fpr, fnr)
    return prior

def calulate_probability_of_evidence(sequence, locations, possible_locations, fpr, fnr):
    return sum(calculate_data_probability(sequence, locations, possible_location, fpr, fnr) for possible_location in possible_locations)
    
#%%
if __name__ == '__main__':
    #%%
    sprt = SequentialProbRatioTest(0.8, 0.2, 0.1)
    sprt.plot_critical_region()    
    sprt.plot_lower_and_upper_decision_boundary_fixed_type2_error(0.1)
    sprt.plot_lower_and_upper_decision_boundary_fixed_type1_error(0.1)
    sprt.plot_critical_region()
    #%%
    #3X3 grid
    no_states = 10
    fpr, fnr = 0.45, 0.1
    
    prior = [0.08 for i in range(9)] + [1 - 0.08*9]
    #if test is greater that upper bound or less than lower bound, terminate
    likelihoods = []
    for i in range(18):
        likelihoods.append(LogLikelihoodCalculator(fpr, fnr, prior))
    
    for i in range(9):
        likelihoods[i].update(1, i)
    
    for i in range(9):
        likelihoods[9+i].update(0, i)
    print('\n\n\n')
    
    for i in range(18):
        print(likelihoods[i].likelihood_product)
        print(likelihoods[i].likelihood_product[:-1].sum())
        print(likelihoods[i].likelihood_product.sum())
        print('\n')
    
#%%
    for i in range(9):
        likelihoods[i].update(1, i)
    
    for i in range(9):
        likelihoods[9+i].update(0, i)
    print('\n\n\n')
    
    for i in range(18):
        print(likelihoods[i].likelihood_product)
        print(likelihoods[i].likelihood_product[:-1].sum())
        print(likelihoods[i].likelihood_product.sum())
        print('\n')
        
    import matplotlib.pyplot as plt
    plt.clf()
    plt.plot([i/100 for i in range(1,99)], [math.log((i/100)/(1-(i/100))) for i in range(1,99)])
    plt.plot([i/100 for i in range(1,99)],[1.09861 for i in range(1,99)], label = 'upper bound')
    plt.plot([i/100 for i in range(1,99)],[-0.6931471805599453 for i in range(1,99)], label = 'lower bound')
    plt.legend()
    #%%
    ls = LogLikelihoodCalculator(fpr, fnr, prior)
    assert np.array_equal(ls.likelihood_product, np.array([prior[i] for i in range(10)]))
    ls.update(1, 1)
    assert np.array_equal(ls.likelihood_product, np.array([0.2, 0.6, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], dtype = np.float64))
    
    #record how many positive observations until acceptance
    sprt = SequentialProbRatioTest(no_states, fpr, fnr)
    print("lower bound: ", sprt.lower_bound)
    print("upper bound: ", sprt.upper_bound)
    i = 0
    while not sprt.accept_source_in_grid():
        print(sprt.get_log_likelihood_ratio())
        print(sprt.log_likelihood_calculator.likelihood_product)
        sprt.update_likelihood(0, 1)
        i+=1
    print("number of observations: ", i)
    print(sprt.get_log_likelihood_ratio())
    print(sprt.get_critical_region())
    
    #%%
    test_grid = UE4Grid(1, 1, Vector3r(0,0), 8, 6)
    #prob of postive reading at non-source location = false alarm = alpha
    false_positive_rate = 0.1
    #prob of negative reading at source location = missed detection = beta
    false_negative_rate = 0.13
    cb_bel_map1 = SingleSourceBinaryBeliefMap(test_grid, [BeliefMapComponent(grid_point, 0.008) for grid_point in test_grid.get_grid_points()], 
                                                             {grid_point: 0.008 for grid_point in test_grid.get_grid_points()}, false_positive_rate, false_negative_rate)

    obs1 = AgentObservation(Vector3r(2,4,0),1, 1, 1234, 'agent1')
    obs2 = AgentObservation(Vector3r(3,4),1, 2, 1235, 'agent1')
    obs3 = AgentObservation(Vector3r(1, 5,0),0, 3, 1237, 'agent1')
    obs4 = AgentObservation(Vector3r(2, 5,0),0, 4, 1238, 'agent1')
    
    #%%
    #make sure user can't accidenally use abstract base class
    try:
        SingleSourceSearchTermination()
        assert False
    except TypeError as e:
        assert True



    cb_bel_map1.get_probability_source_in_grid()
    lower_bound = 0.05
    upper_bound = 0.95
    upper_lower_search_termination = UpperLowerBoundTotalBeliefSearchTermination(lower_bound, upper_bound)
    assert not upper_lower_search_termination._accept_source_in_grid(cb_bel_map1)
    assert not upper_lower_search_termination._accept_source_not_in_grid(cb_bel_map1)

    
    #%%
    cb_bel_map1.update_from_observation(obs1)
    cb_bel_map1.update_from_observation(obs2)
    cb_bel_map1.update_from_observation(obs3)
    cb_bel_map1.update_from_observation(obs4)

    assert not upper_lower_search_termination._accept_source_in_grid(cb_bel_map1)
    assert not upper_lower_search_termination._accept_source_not_in_grid(cb_bel_map1)

    #%%
    
    #series of positive observations
    i = 10
    while not upper_lower_search_termination.should_end_search(cb_bel_map1):
        assert cb_bel_map1.get_probability_source_in_grid() < upper_bound
        print(cb_bel_map1.get_probability_source_in_grid())
        cb_bel_map1.update_from_observation(AgentObservation(Vector3r(2,4,0),1, i, 10+i, 'agent1'))
        i+=1
        
    assert cb_bel_map1.get_probability_source_in_grid() > upper_bound

#%%
    import time
    import random
    cb_bel_map1 = SingleSourceBinaryBeliefMap(test_grid, [BeliefMapComponent(grid_point, 0.08) for grid_point in test_grid.get_grid_points()], 
                                                             {grid_point: 0.08 for grid_point in test_grid.get_grid_points()}, false_positive_rate, false_negative_rate)
    
    i = 0
    while not upper_lower_search_termination.should_end_search(cb_bel_map1):
        time.sleep(1)
        assert cb_bel_map1.get_probability_source_in_grid() > lower_bound
        print(cb_bel_map1.get_probability_source_in_grid())
        #need to generate zero readings at multiple grid locations in order to 
        #ensure belief keeps reducing
        print(cb_bel_map1.get_belief_map_component(Vector3r(int(random.random() * 8),int(random.random() * 7),0)))
        cb_bel_map1.update_from_observation(AgentObservation(Vector3r(int(random.random() * 9),int(random.random() * 7)),0, i, 10+i, 'agent1'))
        i+=1
    print("Took {} negative readings to reduce belief".format(i))
    assert cb_bel_map1.get_probability_source_in_grid() < lower_bound

#%%
    #check if recurrence holds
    import time
    import random
    #i = 0
    #prob of postive reading at non-source location = false alarm = alpha
    false_positive_rate = 0.1
    #prob of negative reading at source location = missed detection = beta
    false_negative_rate = 0.2
    test_grid = UE4Grid(1, 1, Vector3r(0,0), 2, 3)
    set_uniform_prior = 0.08
    cb_bel_map1 = SingleSourceBinaryBeliefMap(test_grid, [BeliefMapComponent(grid_point, set_uniform_prior ) for grid_point in test_grid.get_grid_points()], 
                                                             {grid_point: set_uniform_prior  for grid_point in test_grid.get_grid_points()}, false_positive_rate, false_negative_rate)
    
    number_negative_readings = 4
    for i in range(number_negative_readings):
        #time.sleep(1)
        assert cb_bel_map1.get_probability_source_in_grid() > lower_bound
        #print("Probability source is in the grid: ", cb_bel_map1.get_probability_source_in_grid())
        #need to generate zero readings at multiple grid locations in order to 
        #ensure belief keeps reducing
        print("Belief at grid cell at time step {} is {}".format(i, cb_bel_map1.get_belief_map_component(Vector3r(1,2)).likelihood))
        cb_bel_map1.update_from_observation(AgentObservation(Vector3r(1,2),0, i, 10+i, 'agent1'))
        i+=1
    print("after 4 iterations, belief is: ", cb_bel_map1.get_belief_map_component(Vector3r(1,2)).likelihood)
    

    
    def negative_reading_belief_evolution(no_timesteps, alpha, beta, prior_at_grid_cell):
        numerator = prior_at_grid_cell
        denominator =  prior_at_grid_cell - ((prior_at_grid_cell-1) * (((1-alpha)/beta)**no_timesteps))
        return numerator/denominator
    
    print("Actual value: ", cb_bel_map1.get_belief_map_component(Vector3r(1,2)).likelihood)
    print("Predicted value: ", negative_reading_belief_evolution(number_negative_readings, false_positive_rate,false_negative_rate, set_uniform_prior))
    
    assert math.isclose(cb_bel_map1.get_belief_map_component(Vector3r(1,2)).likelihood, negative_reading_belief_evolution(number_negative_readings, false_positive_rate,false_negative_rate, set_uniform_prior), rel_tol = 0.01)


    #%%
    #Test that search termination criteria for difference between most likely and second most likely is great enough.
    
    lower_bound = 0.05
    upper_bound = 0.85
    upper_lower_search_termination = UpperLowerBoundTotalBeliefSearchTermination(lower_bound, upper_bound)
    belief_difference = 0.6
    individual_grid_cell_search_termination = IndividualGridCellSearchTermination(lower_bound, upper_bound, belief_difference)
    #prob of postive reading at non-source location = false alarm = alpha
    false_positive_rate = 0.1
    #prob of negative reading at source location = missed detection = beta
    false_negative_rate = 0.2
    test_grid = UE4Grid(1, 1, Vector3r(0,0), 5, 3)
    set_uniform_prior = 0.008
    cb_bel_map1 = SingleSourceBinaryBeliefMap(test_grid, [BeliefMapComponent(grid_point, set_uniform_prior ) for grid_point in test_grid.get_grid_points()], 
                                                             {grid_point: set_uniform_prior  for grid_point in test_grid.get_grid_points()}, false_positive_rate, false_negative_rate)
    #%%
    from Utils.ObservationSetManager import ObservationSetManager
    observation_set_manager = ObservationSetManager("agent1")
    #add some positive readings at some other grid cell to test that search doesn't terminate even when 
    #belief in individual grid cell is very high
    for i in range(3):
        cb_bel_map1.update_from_observation(AgentObservation(Vector3r(4,2),1, i, i, 'agent1'))
        observation_set_manager.update_with_observation(AgentObservation(Vector3r(4,2),1, i, i, 'agent1'))
    #make sure that the difference in probabilities is greater than the belief_difference specified

#%%
    assert not upper_lower_search_termination.should_end_search(cb_bel_map1), cb_bel_map1.get_probability_source_in_grid()
    print(cb_bel_map1.get_belief_map_component(Vector3r(4,2)).likelihood)
    print(cb_bel_map1.get_belief_map_component(Vector3r(2,3)).likelihood)
    i = 4
    while not upper_lower_search_termination.should_end_search(cb_bel_map1):
        assert cb_bel_map1.get_probability_source_in_grid() > lower_bound
        print("probability source in grid: ", cb_bel_map1.get_probability_source_in_grid())
        cb_bel_map1.update_from_observation(AgentObservation(Vector3r(2,3),1, i, 10+i, 'agent1'))
        print('\n')
        print(cb_bel_map1.get_belief_map_component(Vector3r(4,2)).likelihood)
        print(cb_bel_map1.get_belief_map_component(Vector3r(2,3)).likelihood)
        observation_set_manager.update_with_observation(AgentObservation(Vector3r(2,3),1, i, 10+i, 'agent1'))
        i+=1
            
    print("probability source in grid: ", cb_bel_map1.get_probability_source_in_grid())
    assert cb_bel_map1.get_belief_map_component(Vector3r(4,2)).likelihood > 0.4    
    assert cb_bel_map1.get_belief_map_component(Vector3r(2,3)).likelihood < 0.9
    #%%
    # what happens is that the location 4,2 gets updated to probability 0.8050314465408805, then 
    # 2,3 gets updated from 0.0015723270440251573 to 0.09155937052932762, which reduces the probability of 4,2 to 0.732474964234621. 
    # This pushes the search belief over the threshold of 0.85 and also pushes the probability difference between the two grid cells 
    # to a value greater than 0.9
    print("Difference between first and second most likely grid components: ", cb_bel_map1.get_belief_map_component(Vector3r(4,2)).likelihood - cb_bel_map1.get_belief_map_component(Vector3r(2,3)).likelihood)
    assert not individual_grid_cell_search_termination.should_end_search(cb_bel_map1, observation_set_manager)
     
    #now update with null observations and check that the difference between the first and second most likely grid locations
    #is within the threshold    
    for i in range(3):
        cb_bel_map1.update_from_observation(AgentObservation(Vector3r(4,2),0, i+50, i+30, 'agent1'))
        observation_set_manager.update_with_observation(AgentObservation(Vector3r(4,2),0, i+50, i+50, 'agent1'))
    #
    print("Difference between first and second most likely grid components: ", cb_bel_map1.get_belief_map_component(Vector3r(2,3)).likelihood - cb_bel_map1.get_belief_map_component(Vector3r(4,2)).likelihood)
    assert individual_grid_cell_search_termination.should_end_search(cb_bel_map1, observation_set_manager)

    #%%

    sequence = [1,0,0,0,1,1,0]
    fpr = 0.1
    fnr = 0.2
    locations = [Vector3r(1,1), Vector3r(1,2), Vector3r(1,3), Vector3r(2,2), Vector3r(1,2), Vector3r(4,4), Vector3r(1,2)]
    more_likely_locations = [Vector3r(1,1), Vector3r(1,2), Vector3r(1,3), Vector3r(2,2), Vector3r(1,2), Vector3r(1,1), Vector3r(1,2)]
    test_grid = UE4Grid(1, 1, Vector3r(0,0), 5, 3)
    likelihood = calculate_data_probability(sequence, locations, Vector3r(1,2), fpr, fnr)
    print(likelihood)
    print(calulate_probability_of_evidence(sequence, locations, test_grid.get_grid_points(), fpr, fnr))
    print(calulate_probability_of_evidence(sequence, more_likely_locations, test_grid.get_grid_points(), fpr, fnr))






