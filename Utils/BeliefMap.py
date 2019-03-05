# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 11:40:27 2018

@author: 13383861
"""

import sys
sys.path.append('..')
import typing
from scipy.ndimage.filters import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
#
from Utils.UE4Grid import UE4Grid
#from AirSimInterface.types import Vector3r
from Utils.Vector3r import Vector3r
from Utils.AgentObservation import AgentObservation#, BinaryAgentObservation
from Utils.Prior import generate_gaussian_prior, generate_uniform_prior

#calculation of posterior distribution is bayes likelihood update formula
#This is the posterior distribution of a single sensor reading, independent of all other sensor readings
#calc_posterior = lambda observation, prior: (prior * observation) / ((prior * observation) + (1-prior)*(1-observation))
def calc_posterior(observation, prior):
    numerator = (prior * observation)
    denominator = ((prior * observation) + (1-prior)*(1-observation))
    #if denominator == 0:
        #this is a quick and easy solution, should use log odds to solve this properly
    #    denominator = 0.0000000000000001
    return numerator / denominator

#This calculates the posterior distribution of detection of a source in the grid to be explored, 
#where there is no independence between grid cells
#need to know sensor sensitivity, prior and probability of false positive/false negative
def _calc_posterior_given_sensor_sensitivity(agent_observation_probability: float, agent_observation_grid_loc: Vector3r, location: Vector3r, alpha: 'prob of false pos', beta: 'prob of false neg', prior):
    '''
    As outlined in A Decision-Making Framework for Control Strategies in Probabilistic Search, all grid cell probabilities are updated 
    whenever an observation is made in any grid cell. This function returns the posterior probability of the source being present in a 
    single grid cell having made an observation (possibly in another grid cell)
    '''
    if agent_observation_probability == 1:
        denominator = ((1-beta) * prior) + (alpha * (1-prior))
        if location == agent_observation_grid_loc:
            numerator = (1-beta) * prior
        else:
            numerator =  alpha * prior
    else:
        #observation 0 for both of these
        denominator = (beta * prior) + ((1-alpha) * (1-prior))
        if location == agent_observation_grid_loc:
            numerator = beta * prior
        else:
            numerator = (1-alpha) * prior
    return numerator/denominator

def _calc_posterior_from_sensor(observation_grid_loc_prior: float, grid_cell_to_update_prior: float, observation_value, observation_cell_loc: Vector3r, update_cell_loc: Vector3r, alpha, beta):
    numerator = grid_cell_to_update_prior
    #ordered by most likely to least likely
    if observation_value == 0 and observation_cell_loc != update_cell_loc:
        numerator = (1 - alpha) * numerator
        denominator = ((1 - alpha) * (1 - observation_grid_loc_prior)) + ((beta) * observation_grid_loc_prior)
        
    elif observation_value == 1 and observation_cell_loc != update_cell_loc:
            numerator = (alpha) * numerator
            denominator = ((alpha) * (1 - observation_grid_loc_prior)) + ((1 - beta) * observation_grid_loc_prior)    
            
    elif observation_value == 1 and observation_cell_loc == update_cell_loc:
        assert grid_cell_to_update_prior == observation_grid_loc_prior
        numerator = (1 - beta) * numerator
        denominator = ((alpha) * (1-grid_cell_to_update_prior)) + ((1 - beta) * grid_cell_to_update_prior)
        
    elif observation_value == 0 and observation_cell_loc == update_cell_loc:
        assert grid_cell_to_update_prior == observation_grid_loc_prior
        numerator = beta * numerator
        denominator = ((1 - alpha) * (1-grid_cell_to_update_prior)) + (beta * grid_cell_to_update_prior)
        
    else:
        raise Exception("Cannot calculate posterior when sensor readings are not 0 or 1. You provided value {}".format(observation_value))
        
    return numerator / denominator

#wrapper method for above given an agent observation rather than a probability and recorded grid location
def calc_posterior_given_sensor_sensitivity(observation_grid_loc_prior: float, grid_cell_to_update_prior: float, observation_value: int, observation_cell_loc: Vector3r, update_cell_loc: Vector3r, alpha: float, beta: float):
    '''
    As outlined in A Decision-Making Framework for Control Strategies in Probabilistic Search, all grid cell probabilities are updated 
    whenever an observation is made in any grid cell. This function returns the posterior probability of the source being present in a 
    single grid cell having made an observation (possibly in another grid cell)
    '''
    return _calc_posterior_from_sensor(observation_grid_loc_prior, grid_cell_to_update_prior, observation_value, observation_cell_loc, update_cell_loc, alpha, beta)

#%%
#A belief map component consists of a grid location and a likelihood
_BeliefMapComponent = typing.NamedTuple('belief_map_component', 
                                [('grid_loc',Vector3r),
                                 ('likelihood', float)]
                                )

#%%
class BeliefMapComponent(_BeliefMapComponent):
    '''A wrapper class of BeliefMapComponent to enforce correct data types'''
    def __new__(cls, grid_loc, likelihood) -> '_BeliefMapComponent':
        #does it make sense to just leave args in the constructor and let the return line handle an incorrect number of args
        args = (grid_loc,likelihood)
#        print("BeliefMapComponenet args: ", args)
#        
#        for value, d_type in zip(args, _BeliefMapComponent.__annotations__.values()):
#            print("Type: ",type(value))
#            if(type(value) is not d_type):
#                print("{type_value} is not {d_type}".format(type_value = type(value), d_type=d_type))
#                print(d_type(value))
#        
#        print("Dtypes: ",[d_type(value) if type(value) is not d_type else value for value, d_type in zip(args, _BeliefMapComponent.__annotations__.values())])
        return super().__new__(cls, *[d_type(value) if type(value) is not d_type else value for value, d_type in zip(args, _BeliefMapComponent.__annotations__.values())])

#%%
def get_posterior_given_obs(observations:list, prior):
    '''For a sequence of observations calculates the posterior probability given a prior.
    observation is a list of probabilities'''
    for observation in observations:
        prior = calc_posterior(observation, prior)
    return prior
        

def get_posterior_given_obsChungBurdick(observation: float, prior: typing.Dict[Vector3r, float], alpha: 'prob of false pos', beta: 'prob of false neg'):
    '''For a given observation calculates the posterior probability distribution given a prior.
    Since a positive reading at a grid location will lower probabilty of source being in other locations, 
    need to update the whole posterior distribution.'''
    #given a single observation
    return calc_posterior_given_sensor_sensitivity(observation, alpha, beta, prior)

def get_lower_confidence_given_obs(observations: list):
    '''Given a list of estimates of theta (probability that source is at a location), returns an lower confidence on 
    what the true value should be. 95% confidence interval. Student distribution used due to small sample size'''
    t_coefficients = [6.314,2.92,2.353,2.132,2.015,1.943,1.895,1.86,1.833,1.812,1.796,1.782,1.771,1.761,1.753,1.746,1.74,1.734,1.729,1.725,1.721,1.717,1.714,1.711,1.708,1.706]
    t = t_coefficients[len(observations)]
    if len(observations) == 0:
        return 0
    else:
        return (sum(observations)/len(observations)) - t * (np.std(observations)/(len(observations)**0.5))

def get_upper_confidence_given_obs(observations: list):
    '''Given a list of estimates of theta (probability that source is at a location), returns an upper confidence on 
    what the true value should be. 95% confidence interval. Student distribution used due to small sample size'''
    t_coefficients = [6.314,2.92,2.353,2.132,2.015,1.943,1.895,1.86,1.833,1.812,1.796,1.782,1.771,1.761,1.753,1.746,1.74,1.734,1.729,1.725,1.721,1.717,1.714,1.711,1.708,1.706]
    t = t_coefficients[len(observations)]
    if len(observations) == 0:
        return 1
    else:
        return (sum(observations)/len(observations)) + t * (np.std(observations)/(len(observations)**0.5))

def get_lower_and_upper_confidence_given_obs(observations: typing.List[AgentObservation]):
    return get_lower_confidence_given_obs(observations), get_upper_confidence_given_obs(observations)

#%%
#######################  Belief map and tests  #######################
class ContinuousBeliefMap:
    '''Based on either discrete belief map or interpolation of continuous measurements'''
    def __init__(self, xmin, xmax, ymin, ymax):
        pass
        
#######################  Belief map and tests  #######################
#A belief map has an agent name (beliefs belong to an agent) consists of belief map components
#Leave this as namedtuple if don't need to define methods
class BeliefMap:
    '''Add method to create a continuous belief map'''
    def __init__(self, grid: UE4Grid, belief_map_components: typing.List[BeliefMapComponent], prior: typing.Dict[Vector3r, float], apply_blur = False):
        '''apply_blur applies a gaussian blue to the belief map on each update to provide a more continuous distribution of belief map values'''
        #self.agent_name = agent_name
        self.grid = grid
        self.belief_map_components = belief_map_components
        self.prior = prior
        self.apply_blur = apply_blur
        
        #create a cache of probability updates based on grid size - e.g. for n observations, 
    
    def get_prior(self)->typing.Dict[Vector3r, float]:
        return self.prior
    
    def get_grid(self)->UE4Grid:
        return self.grid
            
    def get_belief_map_components(self):
        return self.belief_map_components
    
    def set_apply_blur(self, apply_blur):
        self.apply_blur = apply_blur
    
    def set_belief_map_components(self, new_belief_map_components):
        self.belief_map_components = new_belief_map_components
        
    def _get_current_likelihood_at_loc(self, grid_loc):
        return self.get_belief_map_component(grid_loc).likelihood
    
    def update_from_prob(self, grid_loc, obs_prob):
        '''Updates likelihodd at grid location given a grid location and observation'''
        prior_val = self._get_current_likelihood_at_loc(grid_loc)
        self.belief_map_components[self._get_observation_grid_index(grid_loc)] = BeliefMapComponent(grid_loc, get_posterior_given_obs([obs_prob], prior_val))
        if self.apply_blur:
            self.apply_gaussian_blur()
        
    def update_from_observation(self, agent_observation: AgentObservation):
        self.update_from_prob(agent_observation.grid_loc, agent_observation.probability)
        
    def update_from_observations(self, agent_observations: typing.Set[AgentObservation]):
        for observation in agent_observations:
            self.update_from_observation(observation)
    
    def _get_observation_grid_index(self, grid_loc: Vector3r):
        return self.belief_map_components.index(self.get_belief_map_component(grid_loc))
    
    def get_belief_map_component(self, grid_loc):
        if grid_loc in map(lambda belief_map_component: belief_map_component.grid_loc ,self.belief_map_components):
            return next(filter(lambda belief_map_component: belief_map_component.grid_loc == grid_loc, self.belief_map_components))
        else:
            print(list(map(lambda belief_map_component: belief_map_component.grid_loc ,self.belief_map_components)))
            print(grid_loc in list(map(lambda belief_map_component: belief_map_component.grid_loc ,self.belief_map_components)))
            raise Exception("{} is not in the belief map".format(grid_loc))
            
    def save_visualisation(self, filepath):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        X = list(map(lambda coord: coord.x_val, self.grid.get_grid_points()))
        Y = list(map(lambda coord: coord.y_val, self.grid.get_grid_points()))
        Z = [grid_comp.likelihood for grid_comp in self.get_belief_map_components()]
        ax.set_zlim3d(0,1)
        ax.plot_trisurf(X, Y, Z)
        plt.savefig(filepath)
    
    def get_most_likely_component(self):
        '''Returns the belief map component that has the highest likelihood of containing the source'''
        return max([component for component in self.belief_map_components], key = lambda component: component.likelihood)
    
    def get_ith_most_likely_component(self, i):
        '''Returns the belief map component that has the highest likelihood of containing the source'''
        return sorted([component for component in self.belief_map_components], key = lambda component: component.likelihood)[i-1] # i-1 necessary for zero-indexed lists
    
    #experimental - shouldn't need to use this in Chung & Burdick framework since bayesian filtering consists of a prediction and smoothing step already.
    def apply_gaussian_blur(self, blur_radius = None):
        '''Idea is that grid locations are not independent, and the joint distribution should be continuous. Apply a smoother in order to 
        prevent large discontinuities. This is addressed properly in the paper "Learning Occupancy Grids with Forward Models. (http://robots.stanford.edu/papers/thrun.iros01-occmap.pdf)'''
        if not blur_radius:
            blur_radius = int(0.15 * len(self.get_grid().get_grid_points()))
        bel_map_locations = [grid_comp.grid_loc for grid_comp in self.get_belief_map_components()]
        grid_array = np.array([grid_comp.likelihood for grid_comp in self.get_belief_map_components()]).reshape([self.grid.get_no_points_x(),self.grid.get_no_points_y()])
        blurred = gaussian_filter(grid_array, sigma = blur_radius)
        self.set_belief_map_components([BeliefMapComponent(bel_map_location, likelihood) for bel_map_location, likelihood in zip(bel_map_locations, blurred.ravel())])
    
    def __eq__(self, other):
        #check if grids are the same an componenents are the same
        pass
    
    def __str__(self):
        return str({"grid": self.grid,"prior": self.prior,"components": self.belief_map_components})
                    #"agent_name": self.agent_name,
                    
    #some useful class methods for analysis
    #maybe these should eventually go in a different place but for now will keep them here
    #add non-class methods to allow the calculation of relative/absolute change in belief
    #given most recent measurement
    @classmethod
    def relative_change_bel_pos_reading(alpha, beta, prev_belief, prior_at_reading):
        '''Calculates the relatives change in belief given a positive reading'''
        alpha1 = 1 - alpha - beta
        numerator = ((alpha * prev_belief) + (alpha1 * prior_at_reading))
        denominator = ((alpha * prev_belief) + (alpha1 * prior_at_reading * prev_belief))
        return numerator/denominator
    
    @classmethod
    def absolute_change_bel_pos_reading(alpha, beta, prev_belief, prior_at_reading):
        '''Calculates the relatives change in belief given a positive reading'''
        gamma = 1 - alpha - beta
        numerator = (prior_at_reading) * (gamma) * (1 - (prev_belief))
        denominator = alpha + (gamma * prior_at_reading)
        return numerator/denominator
    
    @classmethod
    def relative_change_bel_neg_reading(alpha, beta, prev_belief, prior_at_reading):
        '''Calculates the relatives change in belief given a positive reading'''
        beta1 = 1 - alpha - beta
        numerator = (((1-alpha) * prev_belief) - (beta1 * prior_at_reading))
        denominator = (((1-alpha) * prev_belief) - (beta1 * prior_at_reading * prev_belief))
        return numerator/denominator
    
    @classmethod
    def absolute_change_bel_neg_reading(alpha, beta, prev_belief, prior_at_reading):
        '''Calculates the relatives change in belief given a positive reading'''
        gamma = 1 - alpha - beta
        numerator = (prior_at_reading) * (gamma) * ((prev_belief) - 1)
        denominator = (1 - alpha) - (gamma * prior_at_reading)
        return numerator/denominator
 
    
class ChungBurdickBeliefMap(BeliefMap):
    
    '''
    An occupancy grid with single source based on the paper:
    A Decision-Making Framework for Control Strategies in Probabilistic Search
    '''
    
    def __init__(self, grid: UE4Grid, belief_map_components: typing.List[BeliefMapComponent], prior: typing.Dict[Vector3r, float], alpha: 'prob of false pos', beta: 'prob of false neg', apply_blur = False):
        super().__init__(grid, belief_map_components, prior, apply_blur)
        self.alpha = alpha
        self.beta = beta
        
#    def update_from_observation(self, agent_observation: 'BinaryAgentObservation'):
#        '''Updates all likelihoods based on observation'''
#        for grid_loc in self.grid.get_grid_points():
#            prior_val = self._get_current_likelihood_at_loc(grid_loc)
#            #agent_observation_probability: float, agent_observation_grid_loc: Vector3r, location: Vector3r, alpha: 'prob of false pos', beta: 'prob of false neg', prior):
#            new_belief_value = calc_posterior_given_sensor_sensitivity(agent_observation, grid_loc, self.alpha, self.beta, prior_val)
#            self.belief_map_components[self._get_observation_grid_index(grid_loc)] = BeliefMapComponent(grid_loc, new_belief_value)
#        
#        if self.apply_blur:
#            self.apply_gaussian_blur()
        
        
    def update_from_prob_optimized(self, observation_grid_loc, obs_prob):
        '''Updated probability, optimized using local variables and map'''
        observation_grid_loc_prior = self._get_current_likelihood_at_loc(observation_grid_loc)

        def calc_posterior_given_sensor_sensitivity_inner(grid_loc_to_update):
            return calc_posterior_given_sensor_sensitivity(observation_grid_loc_prior, self.get_belief_map_component(grid_loc_to_update).likelihood, obs_prob, observation_grid_loc, grid_loc_to_update, self.alpha, self.beta)
        
        new_beliefs = map(calc_posterior_given_sensor_sensitivity_inner, self.grid.get_grid_points())
        self.belief_map_components = [BeliefMapComponent(grid_loc_to_update, new_belief_value) for grid_loc_to_update, new_belief_value in zip(self.grid.get_grid_points(), new_beliefs)]
        
    def update_from_prob_optimized_one(self, observation_grid_loc, obs_prob):
        '''Updated probability, optimized using local variables and map'''
        observation_grid_loc_prior = self._get_current_likelihood_at_loc(observation_grid_loc)
        grid_points_array = np.array(self.grid.get_grid_points())
        #likelihoods = np.array([self.get_belief_map_component(grid_loc_to_update).likelihood] for grid_loc_to_update in grid_points_array)
        def calc_posterior_given_sensor_sensitivity_inner(grid_loc_to_update):
            return calc_posterior_given_sensor_sensitivity(observation_grid_loc_prior, self.get_belief_map_component(grid_loc_to_update).likelihood, obs_prob, observation_grid_loc, grid_loc_to_update, self.alpha, self.beta)
        
        #new_beliefs = map(calc_posterior_given_sensor_sensitivity_inner, self.grid.get_grid_points())
        new_beliefs_vectorized = np.vectorize(calc_posterior_given_sensor_sensitivity_inner)
        new_beliefs = new_beliefs_vectorized(grid_points_array)
        #new_beliefs = map(calc_posterior_given_sensor_sensitivity_inner, self.grid.get_grid_points())
        self.belief_map_components = [BeliefMapComponent(grid_loc_to_update, new_belief_value) for grid_loc_to_update, new_belief_value in zip(self.grid.get_grid_points(), new_beliefs.tolist())]
        
    def update_from_prob(self, observation_grid_loc, obs_prob):
        '''
        Updates likelihood at all grid locations given a grid location and observation. Mult-processing this would offer a nice speedup.
        '''
        
        observation_grid_loc_prior = self._get_current_likelihood_at_loc(observation_grid_loc)
        #this is really slow...
        for grid_loc_to_update in self.grid.get_grid_points():
            #prior_val_update_cell = self._get_current_likelihood_at_loc(grid_loc)
            #_calc_posterior_given_sensor_sensitivity(agent_observation_probability: float, agent_observation_grid_loc: Vector3r, location: Vector3r, alpha: 'prob of false pos', beta: 'prob of false neg', prior)
            #new_belief_value = _calc_posterior_given_sensor_sensitivity(obs_prob, observation_grid_loc, grid_loc, self.alpha, self.beta, prior_val_update_cell, prior_val_measurement_cell)
            
            grid_cell_to_update_prior = self.get_belief_map_component(grid_loc_to_update).likelihood
            #this could be made to be more efficient - having some kind of cache that stores in advance the updates to the map given a sequence of observations could
            #be beneficial but might not be possible with non-uniform prior
            new_belief_value = calc_posterior_given_sensor_sensitivity(observation_grid_loc_prior, grid_cell_to_update_prior, obs_prob, observation_grid_loc, grid_loc_to_update, self.alpha, self.beta)
            
            self.belief_map_components[self._get_observation_grid_index(grid_loc_to_update)] = BeliefMapComponent(grid_loc_to_update, new_belief_value)        

        if self.apply_blur:
            self.apply_gaussian_blur()
            
            
    def get_probability_source_in_grid(self):
        return sum([belief_map_component.likelihood for belief_map_component in self.belief_map_components])
        


#%%
class ConfidenceIntervalBeliefMap:
    '''Can be used to calculate upper and lower confidence intervals on parameter estimation'''
    
    def __init__(self, grid: UE4Grid, belief_map_components: typing.List[BeliefMapComponent], apply_blur = False):
        '''apply_blur applies a gaussian blue to the belief map on each update to provide a more continuous distribution of belief map values'''
        self.grid = grid
        self.belief_map_components = belief_map_components
        self.apply_blur = apply_blur
    
    def get_grid(self)->UE4Grid:
        return self.grid
            
    def get_belief_map_components(self):
        return self.belief_map_components
    
    def set_belief_map_components(self, new_belief_map_components):
        self.belief_map_components = new_belief_map_components
        
    def _get_current_likelihood_at_loc(self, grid_loc):
        return self.get_belief_map_component(grid_loc).likelihood
    
    def _get_observation_grid_index(self, grid_loc: Vector3r):
        return self.belief_map_components.index(self.get_belief_map_component(grid_loc))
    
    def get_belief_map_component(self, grid_loc):
        if grid_loc in map(lambda belief_map_component: belief_map_component.grid_loc ,self.belief_map_components):
            return next(filter(lambda belief_map_component: belief_map_component.grid_loc == grid_loc, self.belief_map_components))
        else:
            print(list(map(lambda belief_map_component: belief_map_component.grid_loc ,self.belief_map_components)))
            print(grid_loc in list(map(lambda belief_map_component: belief_map_component.grid_loc ,self.belief_map_components)))
            raise Exception("{} is not in the belief map".format(grid_loc))
            
    #override this to calculate upper confidence interval of theta rather than theta 
    def calculate_upper_confidence_from_observations(self, agent_observations: typing.Set[AgentObservation]):
        for grid_loc in self.grid.get_grid_points():
            grid_loc_obs = []
            for agent_observation in agent_observations:
                if agent_observation.grid_loc == grid_loc:
                    grid_loc_obs.append(agent_observation.probability)
            self.belief_map_components[self._get_observation_grid_index(grid_loc)] = BeliefMapComponent(grid_loc, get_upper_confidence_given_obs(grid_loc_obs))    
        
    def calculate_lower_confidence_from_observations(self, agent_observations: typing.Set[AgentObservation]):
        for grid_loc in self.grid.get_grid_points():
            grid_loc_obs = []
            for agent_observation in agent_observations:
                if agent_observation.grid_loc == grid_loc:
                    grid_loc_obs.append(agent_observation.probability)
            self.belief_map_components[self._get_observation_grid_index(grid_loc)] = BeliefMapComponent(grid_loc, get_lower_confidence_given_obs(grid_loc_obs))    
        
    def get_upper_lower_confidence_interval_from_observations(self, agent_observations: typing.Set[AgentObservation]):
        return self.calculate_lower_confidence_from_observations(agent_observations), self.calculate_upper_confidence_from_observations(agent_observations)
    
#%%
#maybe this should go in contsructor and make regular class
def create_belief_map(grid, prior = {}):
    '''Creates an occupancy belief map for a given observer and a set of grid locations.
    Prior is a mapping of grid_points to probabilities'''
    if not prior:
        #use uniform uninformative prior
        prior = generate_uniform_prior(grid, initial_belief_sum = 1)
    return BeliefMap(grid, [BeliefMapComponent(grid_point, prior[grid_point]) for grid_point in grid.get_grid_points()], prior)
    #return {grid_locs[i]: ObsLocation(grid_locs[i],prior[i], 0, time.time(), observer_name) for i in range(len(grid_locs))}

def create_single_source_belief_map(grid, prior = {}, alpha = 0.2, beta = 0.1):
    '''Creates an occupancy belief map for a given observer and a set of grid locations.
    Prior is a mapping of grid_points to probabilities'''
    if not prior:
        #use uniform uninformative prior
        #in this case want all probabilities to add up to 1/2 to indicate maximum uncertainty
        prior = generate_uniform_prior(grid, initial_belief_sum = 0.5)
    return ChungBurdickBeliefMap(grid, [BeliefMapComponent(grid_point, prior[grid_point]) for grid_point in grid.get_grid_points()], prior, alpha, beta)

def create_confidence_interval_belief_map(grid, prior = {}):
    if not prior: 
        prior = generate_uniform_prior(grid, initial_belief_sum = 1)
    return ConfidenceIntervalBeliefMap(grid, [BeliefMapComponent(grid_point, prior[grid_point]) for grid_point in grid.get_grid_points()])
#%%
def create_belief_map_from_observations(grid: UE4Grid, agent_observations: typing.Set[AgentObservation], agent_belief_map_prior: typing.Dict[Vector3r, float] = {}):
    '''Since the calculation of posterior likelihood is based only on prior and observations (independent of order), updating a belief map component from measurements can be done 
    by the following update formula: 
                                prior * product(over all i observations) observation_i
        ----------------------------------------------------------------------------------------------------------------------
        prior * product(over all i observations) observation_i + (1-prior) * product(over all i observations) (1-observation_i)
    '''
    if agent_belief_map_prior:
        return_bel_map = create_belief_map(grid, agent_belief_map_prior)
    else:
        return_bel_map = create_belief_map(grid)
    #update belief map based on all observations...
    return_bel_map.update_from_observations(agent_observations)
    return return_bel_map

def create_single_source_belief_map_from_observations(grid: UE4Grid, agent_observations: typing.Set[AgentObservation], agent_belief_map_prior: typing.Dict[Vector3r, float] = {}):
    '''Since the calculation of posterior likelihood is based only on prior and observations (independent of order), updating a belief map component from measurements can be done 
    by the following update formula: 
                                prior * product(over all i observations) observation_i
        ----------------------------------------------------------------------------------------------------------------------
        prior * product(over all i observations) observation_i + (1-prior) * product(over all i observations) (1-observation_i)
    '''
    if agent_belief_map_prior:
        return_bel_map = create_single_source_belief_map(grid, agent_belief_map_prior)
    else:
        return_bel_map = create_single_source_belief_map(grid)
    #update belief map based on all observations...
    return_bel_map.update_from_observations(agent_observations)
    return return_bel_map


def create_confidence_interval_map_from_observations(grid: UE4Grid, agent_observations: typing.Set[AgentObservation], agent_belief_map_prior: typing.Dict[Vector3r, float] = {}):
    #cannot have prior in this case since CI map relies on MLE
    return_bel_map = create_confidence_interval_belief_map(grid)
    #update belief map based on all observations...
    return_bel_map.update_from_observations(agent_observations)
    return return_bel_map


#%%
if __name__ == "__main__":

#    #A belief map component consists of a grid location and a likelihood
#    _BeliefMapComponent = typing.NamedTuple('belief_map_component', 
#                                [('grid_loc',Vector3r),
#                                 ('likelihood', float)]
#                                )
#    class BeliefMapComponent(_BeliefMapComponent):
#        '''A wrapper class of BeliefMapComponent to enforce correct data types'''
#        def __new__(cls, grid_loc, likelihood) -> '_BeliefMapComponent':
#            #does it make sense to just leave args in the constructor and let the return line handle an incorrect number of args
#            args = (grid_loc,likelihood)
#            return super().__new__(cls, *[d_type(value) if type(value) is not d_type else value for value, d_type in zip(args, _BeliefMapComponent.__annotations__.values())])

    #%%
    test_grid_loc = Vector3r(10,20)
    test_grid_loc1 = Vector3r(float(10.0),float(20.0))
    test_grid_loc2 = Vector3r(float(10),float(20), 0)
    #check that belief map component correctly unpacks values with correct data types
    x = BeliefMapComponent(test_grid_loc, 0.3)
    y = BeliefMapComponent(test_grid_loc1, 0.3)
    z = BeliefMapComponent(test_grid_loc2, 0.3)
    assert x.grid_loc == Vector3r(float(10), float(20))
    assert y.grid_loc == Vector3r(float(10), float(20))
    assert z.grid_loc == Vector3r(float(10), float(20))
    
    #tests for calc_posterior
    assert abs(calc_posterior(0.5, 0.2) - 0.2) <= 0.001
    assert abs(calc_posterior(0.8, 0.2) - 0.5) <= 0.001
    
    print(get_posterior_given_obs([0.4,0.7,0.93], 0.5))
    #tests for get_posterior_given_obs
    assert abs(get_posterior_given_obs([0.5,0.2,0.8], 0.5) - 0.5) <= 0.001
    
    #%%
    #tests for belief map
    test_grid = UE4Grid(1, 1, Vector3r(0,0), 10, 6)
    test_map = create_belief_map(test_grid)
    
    assert all([prior_val == 1/len(test_grid.get_grid_points()) for prior_val in test_map.prior.values()])
    assert test_map.get_belief_map_component(Vector3r(0,0)) == BeliefMapComponent(Vector3r(0,0), 1/len(test_grid.get_grid_points()))
    
    test_map.update_from_prob(Vector3r(0,0), 0.9)
    assert test_map.get_belief_map_component(Vector3r(0,0)).likelihood == calc_posterior(1/len(test_grid.get_grid_points()), 0.9)
    
    #%%
    del test_map
    #prove order in which observations come in doesn't matter
    obs1 = AgentObservation(Vector3r(1.0,2,0),0.4, 1, 1234, 'agent1')
    obs2 = AgentObservation(Vector3r(1,2),0.7, 2, 1235, 'agent1')
    obs3 = AgentObservation(Vector3r(1,2.0,0),0.93, 3, 1237, 'agent1')
    test_map = create_belief_map(test_grid)
    
    assert obs1.grid_loc == Vector3r(float(1), float(2))
    assert test_map.get_belief_map_component(Vector3r(1.0,2,0)).likelihood == 1/len(test_grid.get_grid_points())
    test_map.update_from_observation(obs1)
    assert test_map.get_belief_map_component(Vector3r(1,2)).likelihood == calc_posterior(obs1.probability, 1/len(test_grid.get_grid_points()))

    obs2_prior = test_map.get_belief_map_component(Vector3r(1,2)).likelihood
    test_map.update_from_observation(obs2)
    assert test_map.get_belief_map_component(Vector3r(1,2)).likelihood == calc_posterior(obs2.probability, obs2_prior)
    
    #%%
    del obs2_prior, obs3_prior, test_map
    #%%
    #now check observing in a different order gives same result
    test_map = create_belief_map(test_grid)
    test_map.update_from_observation(obs2)
    test_map.update_from_observation(obs1)
    obs3_prior = test_map.get_belief_map_component(Vector3r(1,2)).likelihood
    test_map.update_from_observation(obs3)
    assert test_map.get_belief_map_component(Vector3r(1,2)).likelihood == calc_posterior(obs3.probability, obs3_prior)
    
    #now check observing in a different order gives same result
    test_map = create_belief_map(test_grid)
    test_map.update_from_observation(obs3)
    test_map.update_from_observation(obs2)
    obs1_prior = test_map.get_belief_map_component(Vector3r(1,2)).likelihood
    test_map.update_from_observation(obs1)
    assert test_map.get_belief_map_component(Vector3r(1,2)).likelihood == calc_posterior(obs1.probability, obs1_prior)
    
    #now check observing in a different order gives same result
    test_map = create_belief_map(test_grid)
    test_map.update_from_observation(obs1)
    test_map.update_from_observation(obs3)
    obs2_prior = test_map.get_belief_map_component(Vector3r(1,2)).likelihood
    test_map.update_from_observation(obs2)
    assert test_map.get_belief_map_component(Vector3r(1,2)).likelihood == calc_posterior(obs2.probability, obs2_prior)
    
    #%%
    unif_prior = generate_uniform_prior(test_grid)
    test_map = create_belief_map(test_grid, unif_prior)

    del test_map
    gaussian_prior = generate_gaussian_prior(test_grid, [1,3], [[7,0], [0,15]], 0.5)
    test_map = create_belief_map(test_grid, gaussian_prior)
#%%
    #now check observing in a different order gives same result
    del test_map
    test_map = create_belief_map(test_grid)
    test_map.update_from_observations([obs3, obs2, obs1])
    assert test_map.get_belief_map_component(Vector3r(1,2)).likelihood == calc_posterior(obs1.probability, calc_posterior(obs2.probability,calc_posterior(obs3.probability, list(test_map.prior.values())[0])))

    del test_map
    #obs4 = AgentObservation(Vector3r(0,2,0),0.4, 1, 1234, 'agent2')
    test_map = create_belief_map_from_observations(test_grid, [obs1, obs2, obs3])
    print(test_map)
    assert 0.2594 < test_map.get_belief_map_component(Vector3r(1,2)).likelihood < 0.2595
    test_map.save_visualisation("C:/Users/13383861/Downloads/test_plot_before_smoothing.png")
    print([comp.likelihood for comp in test_map.get_belief_map_components()])
    test_map.apply_gaussian_blur()
    print([comp.likelihood for comp in test_map.get_belief_map_components()])
    test_map.save_visualisation("C:/Users/13383861/Downloads/test_plot_after_smoothing.png")

#%%
    del test_map
    del test_grid
    test_grid = UE4Grid(1, 1, Vector3r(0,0), 20, 15)
    test_map = create_belief_map(test_grid)
    obs1 = AgentObservation(Vector3r(10,7,0),0.4, 1, 1234, 'agent1')
    obs2 = AgentObservation(Vector3r(10,7),0.7, 2, 1235, 'agent1')
    obs3 = AgentObservation(Vector3r(10, 7,0),0.93, 3, 1237, 'agent1')
    test_map.update_from_observations([obs1, obs2, obs3])    
    
#%%
    print(get_upper_confidence_given_obs([obs1.probability, obs2.probability, obs3.probability]))
    print(get_lower_confidence_given_obs([obs1.probability, obs2.probability, obs3.probability]))
    #BeliefMap(agent_name, grid, [BeliefMapComponent(grid_point, prior[grid_point]) for grid_point in grid.get_grid_points()], prior)
    uCIBelMap = ConfidenceIntervalBeliefMap(test_grid, [BeliefMapComponent(grid_point, 0.1) for grid_point in test_grid.get_grid_points()])
    uCIBelMap.calculate_upper_confidence_from_observations([obs1, obs2, obs3])
    uCIBelMap._get_current_likelihood_at_loc(obs1.grid_loc)
    uCIBelMap.calculate_lower_confidence_from_observations([obs1, obs2, obs3])
    uCIBelMap._get_current_likelihood_at_loc(obs1.grid_loc)
#%%
    test_grid = UE4Grid(1, 1, Vector3r(0,0), 8, 6)
    #prob of postive reading at non-source location = false alarm = alpha
    false_positive_rate = 0.2
    #prob of negative reading at source location = missed detection = beta
    false_negative_rate = 0.1
    cb_bel_map1 = ChungBurdickBeliefMap(test_grid, [BeliefMapComponent(grid_point, 0.008) for grid_point in test_grid.get_grid_points()], 
                                                             {grid_point: 0.008 for grid_point in test_grid.get_grid_points()}, false_positive_rate, false_negative_rate)
    cb_bel_map2 = ChungBurdickBeliefMap(test_grid, [BeliefMapComponent(grid_point, 0.008) for grid_point in test_grid.get_grid_points()], 
                                                             {grid_point: 0.008 for grid_point in test_grid.get_grid_points()}, false_positive_rate, false_negative_rate)
    cb_bel_map3 = ChungBurdickBeliefMap(test_grid, [BeliefMapComponent(grid_point, 0.008) for grid_point in test_grid.get_grid_points()], 
                                                             {grid_point: 0.008 for grid_point in test_grid.get_grid_points()}, false_positive_rate, false_negative_rate)
    #%%prove that total belief is same for single source no matter what order beliefs are in.
    
    #first make sure priors are same
    import math    
    assert math.isclose(cb_bel_map3.get_probability_source_in_grid(),cb_bel_map2.get_probability_source_in_grid(), rel_tol = 0.001) and math.isclose(cb_bel_map3.get_probability_source_in_grid() ,cb_bel_map2.get_probability_source_in_grid(), rel_tol = 0.001) 

    #%%
    obs1 = AgentObservation(Vector3r(2,4,0),1, 1, 1234, 'agent1')
    obs2 = AgentObservation(Vector3r(3,4),1, 2, 1235, 'agent1')
    obs3 = AgentObservation(Vector3r(1, 5,0),0, 3, 1237, 'agent1')
    obs4 = AgentObservation(Vector3r(2, 5,0),0, 4, 1238, 'agent1')
    
    cb_bel_map1.update_from_observation(obs1)
    cb_bel_map1.update_from_observation(obs2)
    cb_bel_map1.update_from_observation(obs3)
    cb_bel_map1.update_from_observation(obs4)
    
    cb_bel_map2.update_from_observation(obs3)
    cb_bel_map2.update_from_observation(obs2)
    cb_bel_map2.update_from_observation(obs1)
    cb_bel_map2.update_from_observation(obs4)
    
    cb_bel_map3.update_from_observation(obs4)
    cb_bel_map3.update_from_observation(obs3)
    cb_bel_map3.update_from_observation(obs1)
    cb_bel_map3.update_from_observation(obs2)

    #now make sure that the order in which updates are made doesn't matter
    assert math.isclose(cb_bel_map3.get_probability_source_in_grid(),cb_bel_map2.get_probability_source_in_grid(), rel_tol = 0.001) and math.isclose(cb_bel_map3.get_probability_source_in_grid() ,cb_bel_map2.get_probability_source_in_grid(), rel_tol = 0.001) 
    
    #%%
    # Check whether the optimized grid map gives same results as unoptimized grid map
    cb_bel_map1 = ChungBurdickBeliefMap(test_grid, [BeliefMapComponent(grid_point, 0.008) for grid_point in test_grid.get_grid_points()], 
                                                             {grid_point: 0.008 for grid_point in test_grid.get_grid_points()}, false_positive_rate, false_negative_rate)
    cb_bel_map2 = ChungBurdickBeliefMap(test_grid, [BeliefMapComponent(grid_point, 0.008) for grid_point in test_grid.get_grid_points()], 
                                                             {grid_point: 0.008 for grid_point in test_grid.get_grid_points()}, false_positive_rate, false_negative_rate)
    cb_bel_map3 = ChungBurdickBeliefMap(test_grid, [BeliefMapComponent(grid_point, 0.008) for grid_point in test_grid.get_grid_points()], 
                                                             {grid_point: 0.008 for grid_point in test_grid.get_grid_points()}, false_positive_rate, false_negative_rate)

    cb_bel_map1.update_from_prob_optimized(Vector3r(0,0), 1)
    cb_bel_map1.update_from_prob_optimized(Vector3r(0,1), 0)
    cb_bel_map1.update_from_prob_optimized(Vector3r(0,1), 1)
    cb_bel_map1.update_from_prob_optimized(Vector3r(0,2), 0)
    
    cb_bel_map2.update_from_prob(Vector3r(0,0), 1)
    cb_bel_map2.update_from_prob(Vector3r(0,1), 0)
    cb_bel_map2.update_from_prob(Vector3r(0,1), 1)
    cb_bel_map2.update_from_prob(Vector3r(0,2), 0)
    
    cb_bel_map3.update_from_prob_optimized_one(Vector3r(0,0), 1)
    cb_bel_map3.update_from_prob_optimized_one(Vector3r(0,1), 0)
    cb_bel_map3.update_from_prob_optimized_one(Vector3r(0,1), 1)
    cb_bel_map3.update_from_prob_optimized_one(Vector3r(0,2), 0)
    
    
    assert math.isclose(cb_bel_map2.get_probability_source_in_grid(),cb_bel_map1.get_probability_source_in_grid(), rel_tol = 0.0001) and math.isclose(cb_bel_map2.get_probability_source_in_grid(),cb_bel_map3.get_probability_source_in_grid(), rel_tol = 0.0001)
    for comp1, comp2, comp3 in zip(cb_bel_map1.get_belief_map_components(), cb_bel_map2.get_belief_map_components(), cb_bel_map3.get_belief_map_components()):
        assert math.isclose(comp1.likelihood, comp2.likelihood, rel_tol = 0.0001)
        assert math.isclose(comp2.likelihood, comp3.likelihood, rel_tol = 0.0001)
        



    
    #%%
    
    #interested to know how much is lost and gained by each grid cell given a positive/negative reading
    #In theory, the relative increase in belief when a positive observation is made is:
    #(assume beta = alpha + alpha1)
    #assume positive reading was made at k
    #Bel(t) = (alpha1 * prior(k) + alpha * Bel(t-1)) /  ((alpha + alpha1 * prior(k)) * Bel(t-1))
    
    #%%
    #these functions give the values for the relative change in belief
    def relative_change_bel_pos_reading(alpha, beta, prev_belief, prior_at_reading):
        '''Calculates the relatives change in belief given a positive reading'''
        alpha1 = 1 - alpha - beta
        numerator = ((alpha * prev_belief) + (alpha1 * prior_at_reading))
        denominator = ((alpha * prev_belief) + (alpha1 * prior_at_reading * prev_belief))
        return numerator/denominator
    
    def absolute_change_bel_pos_reading(alpha, beta, prev_belief, prior_at_reading):
        '''Calculates the relatives change in belief given a positive reading'''
        gamma = 1 - alpha - beta
        numerator = (prior_at_reading) * (gamma) * (1 - (prev_belief))
        denominator = alpha + (gamma * prior_at_reading)
        return numerator/denominator
    
    def relative_change_bel_neg_reading(alpha, beta, prev_belief, prior_at_reading):
        '''Calculates the relatives change in belief given a positive reading'''
        beta1 = 1 - alpha - beta
        numerator = (((1-alpha) * prev_belief) - (beta1 * prior_at_reading))
        denominator = (((1-alpha) * prev_belief) - (beta1 * prior_at_reading * prev_belief))
        return numerator/denominator
    
    def absolute_change_bel_neg_reading(alpha, beta, prev_belief, prior_at_reading):
        '''Calculates the relatives change in belief given a positive reading'''
        gamma = 1 - alpha - beta
        numerator = (prior_at_reading) * (gamma) * ((prev_belief) - 1)
        denominator = (1 - alpha) - (gamma * prior_at_reading)
        return numerator/denominator
    
    #%%
    test_grid = UE4Grid(1, 1, Vector3r(0,0), 8, 6)
    #prob of postive reading at non-source location = false alarm = alpha
    false_positive_rate = 0.2
    #prob of negative reading at source location = missed detection = beta
    false_negative_rate = 0.1
    cb_bel_map = ChungBurdickBeliefMap(test_grid, [BeliefMapComponent(grid_point, 0.008) for grid_point in test_grid.get_grid_points()], 
                                                             {grid_point: 0.008 for grid_point in test_grid.get_grid_points()}, false_positive_rate, false_negative_rate)
    
    import math
    prev_belief = cb_bel_map.get_probability_source_in_grid() 
    prior_val_at_reading = cb_bel_map.get_belief_map_component(Vector3r(0,0)).likelihood
    cb_bel_map.update_from_prob(Vector3r(0,0), 1)
    current_belief = cb_bel_map.get_probability_source_in_grid()
    #numerator = ((1-false_negative_rate) * prior_val) + (false_positive_rate*(prev_bel-prior_val))
    #denominator = ((false_positive_rate * (1-prior_val)) + ((1-false_negative_rate) * (prior_val))) * prev_bel
    assert math.isclose(current_belief/prev_belief, relative_change_bel_pos_reading(false_positive_rate, false_negative_rate, prev_belief, prior_val_at_reading), rel_tol = 0.001)
    assert math.isclose(current_belief-prev_belief, absolute_change_bel_pos_reading(false_positive_rate, false_negative_rate, prev_belief, prior_val_at_reading), rel_tol = 0.001)
    
    #%%
    #check holds for another positive reading
    prev_belief = cb_bel_map.get_probability_source_in_grid() 
    prior_val_at_reading = cb_bel_map.get_belief_map_component(Vector3r(0,4)).likelihood
    cb_bel_map.update_from_prob(Vector3r(0,4), 1)
    current_belief = cb_bel_map.get_probability_source_in_grid()
    assert math.isclose(current_belief/prev_belief, relative_change_bel_pos_reading(false_positive_rate, false_negative_rate, prev_belief, prior_val_at_reading), rel_tol = 0.001)
    assert math.isclose(current_belief-prev_belief, absolute_change_bel_pos_reading(false_positive_rate, false_negative_rate, prev_belief, prior_val_at_reading), rel_tol = 0.001)
    
    
    #%%
    
    #now check given a negative reading, the belief map updates as it should.
    prev_bel = cb_bel_map.get_probability_source_in_grid() 
    prior_val_at_reading = cb_bel_map.get_belief_map_component(Vector3r(2,0)).likelihood
    cb_bel_map.update_from_prob(Vector3r(2,0), 0)
    current_bel = cb_bel_map.get_probability_source_in_grid()
    assert math.isclose(current_bel/prev_bel, relative_change_bel_neg_reading(false_positive_rate, false_negative_rate, prev_belief, prior_val_at_reading), rel_tol = 0.001)
    print((relative_change_bel_neg_reading(false_positive_rate, false_negative_rate, prev_belief, prior_val_at_reading) - 1) * prev_bel)
    #due to round-off, can only get to within 2%
    assert math.isclose(current_bel-prev_bel, absolute_change_bel_neg_reading(false_positive_rate, false_negative_rate, prev_belief, prior_val_at_reading), rel_tol = 0.02)

#%%
    #check holds for another negative reading
    prev_bel = cb_bel_map.get_probability_source_in_grid() 
    prior_val_at_reading = cb_bel_map.get_belief_map_component(Vector3r(2,0)).likelihood
    cb_bel_map.update_from_prob(Vector3r(2,0), 0)
    current_bel = cb_bel_map.get_probability_source_in_grid()
    assert math.isclose(current_bel/prev_bel, relative_change_bel_neg_reading(false_positive_rate, false_negative_rate, prev_belief, prior_val_at_reading), rel_tol = 0.001)
    #due to round-off, can only get to within 2%
    assert math.isclose(current_bel-prev_bel, absolute_change_bel_neg_reading(false_positive_rate, false_negative_rate, prev_belief, prior_val_at_reading), rel_tol = 0.02)
    
    #%%
    
    #the plot of relative change in belief looks like this:
    import matplotlib.pyplot as plt
    
    
    for j in range(1,10):
        plt.plot([1/i for i in range(1, 100)], [relative_change_bel_pos_reading(false_positive_rate,false_negative_rate, 1/i, j/10) for i in range(1,100)], label = "Prior value at observation={}".format(j/10))
    plt.title("Relative belief increase for a positive reading vs prior belief (alpha = 0.2, beta = 0.1, prior at observation location = 0.1")
    #plt.ylim(0, 50)
    plt.xlabel("Prior belief")
    plt.ylabel("Relative change in belief")
    plt.legend()
    
    plt.figure()
    for j in range(1,10):
        plt.plot([1/i for i in range(1, 100)], [relative_change_bel_neg_reading(false_positive_rate,false_negative_rate, 1/i, j/10) for i in range(1,100)], label = "Prior value at observation={}".format(j/10))
    plt.title("Relative belief increase for a positive reading vs prior belief (alpha = 0.2, beta = 0.1, prior at observation location = 0.1")
    #plt.ylim(0, 50)
    plt.xlabel("Prior belief")
    plt.ylabel("Relative change in belief")
    plt.legend()
    #write some assertions here!
    
#%%
    import math
    import random
    import time
    import matplotlib.pyplot as plt
    timings = []
    #check how long updated a belief map takes as a function of grid size
    #grid sizes are from 100 to 2704
    no_grids = 12
    no_updates = 10
    test_grids = [UE4Grid(1, 1, Vector3r(0,0), 12 + i*3, 12 + i*3) for i in range(no_grids)]
    false_positive_rate = 0.2
    false_negative_rate = 0.1
    for test_grid in test_grids:
        t1 = time.time
        bel_map = ChungBurdickBeliefMap(test_grid, [BeliefMapComponent(grid_point, 0.002) for grid_point in test_grid.get_grid_points()], 
                                                             {grid_point: 0.002 for grid_point in test_grid.get_grid_points()}, false_positive_rate, false_negative_rate)
        t1 = time.time()
        
        for i in range(no_updates):
            bel_map.update_from_prob_optimized_one(Vector3r(2+i, 3+i), round(random.random()))
        t2 = time.time()
        timings.append((t2 - t1)/no_updates)
    
    plt.figure()
    plt.plot([(12 + i*3)**2 for i in range(no_grids)],timings)
    plt.xlabel("Size of belief map grid")
    plt.ylabel("Time taken per update")


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        