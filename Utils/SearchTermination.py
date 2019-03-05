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
from Utils.BeliefMap import ChungBurdickBeliefMap
from Utils.ObservationSetManager import ObservationSetManager, get_lower_and_upper_confidence_given_obs

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

class SingleSourceSearchTermination:
    '''Superclass of other classes that determine when to terminate search'''
    def __init__(self):
        pass
    
    def should_end_search(self, belief_map) -> bool:
        '''Given a belief map, returns true if the search should be terminated or false if the search should continue.'''
        if not isinstance(belief_map, ChungBurdickBeliefMap):
            raise NotImplementedError("Calculating the log likelihood ratio is only implemented for the single source framework")
        else:
            raise NotImplementedError("This is a base class - use a subclass")
    
class UpperLowerBoundTotalBeliefSearchTermination(SingleSourceSearchTermination):
    '''A class which terminates the search if the total belief exceeds a user-specified upper bound or a lower bound'''
    def __init__(self, upper_belief_bound, lower_belief_bound):
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
            second_greatest_likelihood_component = belief_map.get_most_likely_component()
            return greatest_likelihood_component.likelihood - second_greatest_likelihood_component.likelihood > self.min_belief_difference
        
        else:
            return False
        

class SequentialProbRatioTest(SingleSourceSearchTermination):
    '''A class which terminates the search according to Wald's Sequential Likelihood Ratio Test'''
    
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
            return math.log((1-bel_map.get_probability_source_in_grid())/bel_map.get_probability_source_in_grid())
    
    def calculate_next_s_i(self, s_i):
        next_s_i = s_i + log_likelihood_ratio(s_i)
        return next_s_i
    
    

if __name__ == '__main__':
    pass











