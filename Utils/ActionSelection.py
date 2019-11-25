# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 11:55:04 2018

@author: 13383861
"""

import random
import sys
#update path so other modules can be imported
sys.path.append('..')


from sklearn.cluster import KMeans
import numpy as np
from abc import ABC, abstractmethod

from Utils.AgentObservation import AgentObservation
from Utils.ObservationSetManager import ObservationSetManager
#from AirSimInterface.types import Vector3r
from Utils.Vector3r import Vector3r
from Utils.UE4Grid import UE4Grid

#create_belief_map,
from Utils.TSPLib import TSPSolver    
from Utils.Timer import timed


#%%

class BaseActionSelection(ABC):
    def __init__(self):
        pass
    
    def get_move(self, belief_map, current_grid_loc, explored_grid_locs) -> (str, Vector3r):
        '''Every agent should have a get move method which will override this method'''
        pass

#%%    
class EpsilonGreedyActionSelection(BaseActionSelection):
    
    def __init__(self, epsilon, eff_radius):
        self.epsilon = epsilon
        self.eff_radius = eff_radius
      
    def get_move_from_belief_map_epsilon_greedy(self, belief_map, current_grid_loc: Vector3r, epsilon: float, eff_radius = None) -> (bool, Vector3r):
        '''Epsilon greedy move selection based on neighbors in belief map'''
        
        #assume grid is regular, get all neighbors that are within the rectangle max(lat_spacing, long_spacing)
        #assuming that lat_spacing < 2* lng_spacing and visa versa
        
        #eff radius is the radius in which the agent can explore - only let it move to neighbors within a certain radius
        if not eff_radius:
            eff_radius = max(belief_map.get_grid().get_lat_spacing(), belief_map.get_grid().get_lng_spacing())
        #a list of Vector3r
        neighbors = belief_map.get_grid().get_neighbors(current_grid_loc, eff_radius)
        #don't move to new position if can't find any neighbors to move to
        if not neighbors:
            return current_grid_loc
        #neighbors = list(filter(lambda grid_loc: grid_loc.get_dist_to_other(current_grid_loc) <= eff_radius and grid_loc!=current_grid_loc, bel_map.keys()))
        greedy = False
        if random.random() < epsilon:
            #return a non-greedy random move
            return_move = random.choice(neighbors)
            greedy = True
        else:
            
           return_move = self.get_greedy_move(neighbors, belief_map, current_grid_loc, epsilon, eff_radius)            

        return 'move', return_move
    
    def get_greedy_move(self,neighbors, belief_map, current_grid_loc: Vector3r, epsilon: float, eff_radius = None) -> (bool, Vector3r):
        best_moves = sorted(neighbors, key = lambda neighbor: belief_map.get_belief_map_component(neighbor).likelihood, reverse=True)
        #print('Best moves: ', list([(move_within_radius, belief_map.get_belief_map_component(move_within_radius).likelihood) for move_within_radius in best_moves]))

        #if a move is within this, consider it a negligible difference. This should be passed through constructor
        negligible = 0.0001
        #of all moves that are within a neglible distance of the best, pick the closest
        moves_within_radius = list(filter(lambda poss_move: belief_map.get_belief_map_component(poss_move).likelihood + negligible > belief_map.get_belief_map_component(best_moves[0]).likelihood, best_moves))
        
        #print('Best moves by likelihood: ', list([(move_within_radius, belief_map.get_belief_map_component(move_within_radius).likelihood) for move_within_radius in moves_within_radius]))

        #if they are of equal closeness, pick randomly?
        return_moves = sorted(moves_within_radius, key = lambda neighbor: neighbor.distance_to(current_grid_loc))
        #print('Best moves ordered by proximity: ', return_moves)
        return_move = return_moves[0]
        return return_move
    
    def get_move(self, belief_map, current_grid_loc: Vector3r, explored_grid_locs: 'list of Vector3r') -> (bool, Vector3r):
        return self.get_move_from_belief_map_epsilon_greedy(belief_map, current_grid_loc, self.epsilon, self.eff_radius)
            
#%%
class GreedyActionSelection(EpsilonGreedyActionSelection):
    '''Greedy action selection is the same as epsilon greedy with a value of 1 for epsilon'''
    def __init__(self, eff_radius):
        super().__init__(0, eff_radius)
        
#%%
class TSPActionSelection(BaseActionSelection):
    @timed
    def __init__(self, grid: UE4Grid, start_node = None):
        self.start_node = start_node
        self.grid = grid
        self.tsp_solver = TSPSolver(self.grid.get_grid_points(), dist_calculator = lambda coord1, coord2: coord1.distance_to(coord2))
        self.moves = self.tsp_solver.ensemble_tsp()
        #self.moves = self.tsp_solver.nn_tsp(start = self.start_node)
        self.move_iterator = iter(self.moves)

    def get_moves(self):
        return self.moves

    def get_next_move(self):
        '''Uses solution to travellling salesman to create a plan that ignores sensor data and simply traverses the region in as 
        efficient a manner as possible with respect to distance'''
        try:
            return next(self.move_iterator)
        #in the case that there is nothing more to explore 
        except StopIteration:
            #reverse the move iterator and get the agent to backtrack
            #maybe it makes more sense to just terminate ?
            self.moves.reverse()
            self.move_iterator = iter(self.moves)
            return next(self.move_iterator)
        
    def get_move(self, belief_map, current_grid_loc, explored_grid_locs):
        #no greedy moves so always return false
        return 'move', self.get_next_move()

#%%
class TSPNNActionSelection(BaseActionSelection):
    @timed
    def __init__(self, grid: UE4Grid, start_node = None):
        self.start_node = start_node
        self.grid = grid
        self.tsp_solver = TSPSolver(self.grid.get_grid_points(), dist_calculator = lambda coord1, coord2: coord1.distance_to(coord2))
        self.moves = self.tsp_solver.nn_tsp(start_node)
        assert len(self.moves) == len(self.grid.get_grid_points())
        assert self.moves[0] == start_node
        #self.moves = self.tsp_solver.nn_tsp(start = self.start_node)
        self.move_iterator = iter(self.moves)

    def get_moves(self):
        return self.moves

    def get_next_move(self):
        '''Uses solution to travellling salesman to create a plan that ignores sensor data and simply traverses the region in as 
        efficient a manner as possible with respect to distance'''
        try:
            return next(self.move_iterator)
        #in the case that there is nothing more to explore 
        except StopIteration:
            #reverse the move iterator and get the agent to backtrack
            #maybe it makes more sense to just terminate ?
            self.moves.reverse()
            self.move_iterator = iter(self.moves)
            return next(self.move_iterator)
        
    def get_move(self, belief_map, current_grid_loc, explored_grid_locs):
        #no greedy moves so always return false
        return 'move', self.get_next_move()    

#%%
class TSPActionSelectionWithPrior(BaseActionSelection):
    '''If the agent has a prior distribution of each grid location, provide this to the 
    TSP solver so that the calculated route visits "high risk" locations earlier '''
    
    def __init__(self, grid: UE4Grid, start_node, prior = {}):
        #if prior not specified or not valid, create uniform prior
        self.start_node = start_node
        self.grid = grid
        #how to construct a distance/cost function that takes the prior into account.
        self.tsp_solver = TSPSolver(self.grid.get_grid_points(), dist_calculator = lambda coord1, coord2: coord1.distance_to(coord2)*prior[coord1]*prior[coord2])
        self.moves = self.tsp_solver.ensemble_tsp()
        #self.moves = self.tsp_solver.nn_tsp(start = self.start_node)
        self.move_iterator = iter(self.moves)

    def get_moves(self):
        return self.moves

    def get_next_move(self):
        '''Uses solution to travellling salesman to create a plan that ignores sensor data and simply traverses the region in as 
        efficient a manner as possible with respect to distance'''
        try:
            return next(self.move_iterator)
        #in the case that there is nothing more to explore 
        except StopIteration:
            #reverse the move iterator and get the agent to backtrack
            #maybe it makes more sense to just terminate ?
            self.moves.reverse()
            self.move_iterator = iter(self.moves)
            return next(self.move_iterator)
        
    def get_move(self, belief_map, current_grid_loc, explored_grid_locs):
        #no greedy moves so always return false
        return 'move', self.get_next_move()
    
#%%

class SaccadicActionSelection(GreedyActionSelection):
    '''
    As outlined in:
    A Decision-Making Framework for Control Strategies in Probabilistic Search.
    '''
    def __init__(self, grid: UE4Grid):
        super().__init__(grid.get_diameter())

    def get_move(self, belief_map, current_grid_loc, explored_grid_locs):
        '''Return the largest value in the belief map to explore next'''
        #find the maximum likelihood belief map component
        max_belief_loc = max(belief_map.get_belief_map_components(), key = lambda bel_map_component: bel_map_component.likelihood)
        return 'move', max_belief_loc.grid_loc
    
    
class RandomActionSelection(BaseActionSelection):
    
    def __init__(self, grid: UE4Grid):
        self.grid = grid
    
    def get_move(self, belief_map, current_grid_loc, explored_grid_locs) -> (str, Vector3r):
        '''Every agent should have a get move method which will override this method'''
        return 'move', random.choice(self.grid.get_grid_points())
    
class LookaheadActionSelection(BaseActionSelection):
    '''
    Returns the sequence of actions to take that will provide the best results (under the uncertainty of the model)
    N steps into the future.
    '''
    pass
    
#%%
class EpsilonGreedyActionSelectionWithBattery():
    '''
    Returns an Epsilon Greedy Move, but also taken into account the battery capacity estimated state
    '''
    def __init__(self, epsilon, eff_radius):
        self.epsilon = epsilon
        self.eff_radius = eff_radius
      
    def get_move_from_belief_map_epsilon_greedy(self, belief_map, current_grid_loc: Vector3r, epsilon: float, eff_radius = None) -> (bool, Vector3r):
        '''Epsilon greedy move selection based on neighbors in belief map'''
        
        #assume grid is regular, get all neighbors that are within the rectangle max(lat_spacing, long_spacing)
        #assuming that lat_spacing < 2* lng_spacing and visa versa
        
        #eff radius is the radius in which the agent can explore - only let it move to neighbors within a certain radius
        if not eff_radius:
            eff_radius = max(belief_map.get_grid().get_lat_spacing(), belief_map.get_grid().get_lng_spacing())
        #a list of Vector3r
        neighbors = belief_map.get_grid().get_neighbors(current_grid_loc, eff_radius)
        #don't move to new position if can't find any neighbors to move to
        if not neighbors:
            return current_grid_loc
        #neighbors = list(filter(lambda grid_loc: grid_loc.get_dist_to_other(current_grid_loc) <= eff_radius and grid_loc!=current_grid_loc, bel_map.keys()))
        greedy = False
        if random.random() < epsilon:
            #return a non-greedy random move
            return_move = random.choice(neighbors)
            greedy = True
        else:
            
           return_move = self.get_greedy_move(neighbors, belief_map, current_grid_loc, epsilon, eff_radius)            

        return 'move', return_move
    
    def get_greedy_move(self,neighbors, belief_map, current_grid_loc: Vector3r, epsilon: float, eff_radius = None) -> (bool, Vector3r):
        best_moves = sorted(neighbors, key = lambda neighbor: belief_map.get_belief_map_component(neighbor).likelihood, reverse=True)
        #print('Best moves: ', list([(move_within_radius, belief_map.get_belief_map_component(move_within_radius).likelihood) for move_within_radius in best_moves]))

        #if a move is within this, consider it a negligible difference. This should be passed through constructor
        negligible = 0.0001
        #of all moves that are within a neglible distance of the best, pick the closest
        moves_within_radius = list(filter(lambda poss_move: belief_map.get_belief_map_component(poss_move).likelihood + negligible > belief_map.get_belief_map_component(best_moves[0]).likelihood, best_moves))
        
        #print('Best moves by likelihood: ', list([(move_within_radius, belief_map.get_belief_map_component(move_within_radius).likelihood) for move_within_radius in moves_within_radius]))

        #if they are of equal closeness, pick randomly?
        return_moves = sorted(moves_within_radius, key = lambda neighbor: neighbor.distance_to(current_grid_loc))
        #print('Best moves ordered by proximity: ', return_moves)
        return_move = return_moves[0]
        return return_move
    
    def move_utility(self, belief_map, expected_battery_value, current_grid_loc, next_grid_loc):
        '''
        Returns the utilitiy function associated with moving from current_grid_loc to next_grid_loc
        with the current expected_battery_value and belief_map
        '''
        pass
    
    def get_move(self, belief_map, current_grid_loc: Vector3r, explored_grid_locs: 'list of Vector3r') -> (bool, Vector3r):
        return self.get_move_from_belief_map_epsilon_greedy(belief_map, current_grid_loc, self.epsilon, self.eff_radius)

#%%
if __name__ == "__main__":

    test_grid = UE4Grid(1, 1, Vector3r(0,0), 6, 5)

    tsp = TSPActionSelection(test_grid, Vector3r(0,3))
        
    print(tsp.get_moves())

    
    obs1 = AgentObservation(Vector3r(0,0),0.5, 1, 1234, 'agent2')
    obs2 = AgentObservation(Vector3r(0,0),0.7, 2, 1235, 'agent2')
    obs3 = AgentObservation(Vector3r(0,1),0.95, 3, 1237, 'agent2')
    #(grid, agent_name, prior = {})
    obs_man = ObservationSetManager("agent1")
    obs_man.update_rav_obs_set('agent2', [obs1, obs2, obs3])
    belief_map = obs_man.get_discrete_belief_map_from_observations(test_grid)
    
    epsilon_greedy = EpsilonGreedyActionSelection(0.0, 1.8)
    assert epsilon_greedy.get_move(belief_map, Vector3r(1,1), []) == (False, Vector3r(0,1))
    
    #test that agent will travel further for higher prediction
    test_grid1 = UE4Grid(2, 1, Vector3r(0,0), 6, 5)
    
    obs_man1 = ObservationSetManager("agent1")
    epsilon_greedy1 = EpsilonGreedyActionSelection(0.0, 2.8)
    obs4 = AgentObservation(Vector3r(2, 0),0.95, 4, 1237, 'agent2')
    obs5 = AgentObservation(Vector3r(0, 1),0.9, 5, 1237, 'agent2')
    obs_man1.update_rav_obs_set('agent2', [obs4, obs5])
    belief_map1 = obs_man1.get_discrete_belief_map_from_observations(test_grid)
    assert epsilon_greedy1.get_move(belief_map1, Vector3r(1,1), []) == (False, Vector3r(2,0))
    
    
    
    sac_action_selection = SaccadicActionSelection(test_grid)
    assert sac_action_selection.get_move(belief_map1, Vector3r(6,5), []) == (False,Vector3r(2,0))
    
    #%%

    grid = UE4Grid(1, 1, Vector3r(0,0), 10, 8)
    grid.get_grid_points()
    agent_start_pos = Vector3r(10,8)
    tsp = TSPActionSelection(grid, agent_start_pos)
    moves = tsp.get_moves()
    print(list(map(lambda x: x.x_val, moves)))
    print(list(map(lambda x: x.y_val, moves)))
    import matplotlib.pyplot as plt
    plt.figure()
    
    plt.plot(list(map(lambda x: x.x_val, moves)), list(map(lambda x: x.y_val, moves)))
        

