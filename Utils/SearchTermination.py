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
from abc import ABC, abstractmethod
from Utils.BeliefMap import ChungBurdickBeliefMap, BeliefMapComponent, get_lower_and_upper_confidence_given_obs
from Utils.ObservationSetManager import ObservationSetManager
import sys
sys.path.append('..')

from Utils.UE4Grid import UE4Grid
#from AirSimInterface.types import Vector3r
from Utils.Vector3r import Vector3r
from Utils.AgentObservation import AgentObservation#, BinaryAgentObservation
from Utils.Prior import generate_gaussian_prior, generate_uniform_prior

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
    if not isinstance(bel_map, ChungBurdickBeliefMap):
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
        if not isinstance(belief_map, ChungBurdickBeliefMap):
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
        if not isinstance(belief_map, ChungBurdickBeliefMap):
            raise NotImplementedError("Total belief search termination is only valid for single-source belief maps")
        
        return self._accept_source_in_grid(belief_map) or self._accept_source_not_in_grid(belief_map)
        
    def _accept_source_in_grid(self, belief_map: ChungBurdickBeliefMap) -> bool:
        return belief_map.get_probability_source_in_grid() > self.upper_belief_bound
    
    def _accept_source_not_in_grid(self, belief_map: ChungBurdickBeliefMap) -> bool:
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
    
    
    def should_end_search(self, belief_map: ChungBurdickBeliefMap, observation_set_manager: ObservationSetManager) -> bool:
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
    
    def should_end_search(self, belief_map: ChungBurdickBeliefMap, observation_set_manager: ObservationSetManager) -> bool:
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
    https://en.wikipedia.org/wiki/Sequential_probability_ratio_test
    '''
    def __init__(self, alpha, beta):
        self.s_i = 0
        self.alpha, self.beta = alpha, beta


    def log_likelihood_ratio(self, s_i, bel_map):
        '''
        Here I want to determine the likelihood ratio between the alternative hypothesis and the null hypothesis. Assuming that the null hypothesis is 
        that the source is not present, calculated as L(theta_0 | x) / L(theta_1 | x) = P(H = 0 | x) / P(H = 1 | x) using Chung & Burdick notation
        '''
        if not isinstance(bel_map, ChungBurdickBeliefMap):
            raise NotImplementedError("Calculating the log likelihood ratio is only implemented for the single source framework")
        else:
            raise NotImplementedError("This has not been implemented yet")
            #return math.log((1-bel_map.get_probability_source_in_grid())/bel_map.get_probability_source_in_grid())
    
    def calculate_next_s_i(self, s_i):
        next_s_i = s_i + log_likelihood_ratio(s_i)
        return next_s_i
    
    
#%%
if __name__ == '__main__':
    #%%
    test_grid = UE4Grid(1, 1, Vector3r(0,0), 8, 6)
    #prob of postive reading at non-source location = false alarm = alpha
    false_positive_rate = 0.1
    #prob of negative reading at source location = missed detection = beta
    false_negative_rate = 0.13
    cb_bel_map1 = ChungBurdickBeliefMap(test_grid, [BeliefMapComponent(grid_point, 0.008) for grid_point in test_grid.get_grid_points()], 
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
    cb_bel_map1 = ChungBurdickBeliefMap(test_grid, [BeliefMapComponent(grid_point, 0.08) for grid_point in test_grid.get_grid_points()], 
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
    cb_bel_map1 = ChungBurdickBeliefMap(test_grid, [BeliefMapComponent(grid_point, set_uniform_prior ) for grid_point in test_grid.get_grid_points()], 
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
    cb_bel_map1 = ChungBurdickBeliefMap(test_grid, [BeliefMapComponent(grid_point, set_uniform_prior ) for grid_point in test_grid.get_grid_points()], 
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














