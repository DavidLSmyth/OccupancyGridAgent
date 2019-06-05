# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 16:04:45 2019

@author: 13383861
"""

import sys
sys.path.append('.')
import os
import time

from abc import ABC, abstractmethod
import numpy as np

from OccupancyGrid.Agents.BaseOccupancyGridAgent import BaseGridAgent
from Utils.Vector3r import Vector3r
from Utils.UE4Grid import UE4Grid
from Utils.BeliefMap import BeliefMapComponent
from Utils.BeliefMapVector import BeliefVectorMultipleSources
from Utils.BatteryDBN import DefaultStochasticBatteryHMM

 
class SimpleGridAgent(BaseGridAgent):
    '''
    This agent is responsible for navigating a discrete 2-dimensional grid (assumed to already be mapped) in order to localize
    a source of evidence. The agent is equipped with appropriate sensors to carry out this task. This is a very simple instantiation, 
    the agent does not communicate nor have battery.
    '''
    
    def __init__(self, grid, initial_pos, move_from_bel_map_callable, height, agent_name, occupancy_sensor_simulator, belief_map_class, init_belief_map, search_terminator, other_active_agents = [], comms_radius = 1000, logged = True):
        super().__init__(grid, initial_pos, move_from_bel_map_callable, height, agent_name, occupancy_sensor_simulator, belief_map_class, init_belief_map, other_active_agents, comms_radius, logged)
        #for simple grid agent assume it is in correct grid square
        self.search_terminator = search_terminator
        
    def reset(self):
        pass
    
    def _execute_action(self, action):
        #This should always be move, recharge incorporated in derived agents
        if action[0] == 'move':
            #need to implement this, first have agent coordinate with others to check if they are intending to move to current
            #grid location
#            if action[1] in other_agent_positions: 
#                #this simulates a lower-level collision avoidence agent instructing 
#                #robot to not crash into others
#                return None
            self.previous_pos_intended = self.current_pos_intended
            self.current_pos_intended = action[1]
        else:
            raise Exception("Invalid action requested by {}: {}".format(self.agent_name, action))
        
        
    def move_agent(self, location: Vector3r, other_agent_positions):
        '''
        This sends the commands to the actuators to physically move the agent
        '''
        if location in other_agent_positions: 
            #this simulates a lower-level collision avoidence agent instructing 
            #robot to not crash into others
            return None
        else:
            self._move_agent(location)

#    def coord_with_other(self, other_rav_name):
#        
#        '''
#        Coordinate with other rav by requesting their measurement list and sending our own measurement list first write own measurement list to file.
#        '''
#        
#        log_msg_to_file(str(other_rav_name) + ' ' + str(self.timestep), "info", "comms", self._logged)
#        self.log_msg_to_cmd("Coordinating with: " + other_rav_name, "debug", "cmd_line", self._logged)
#        observations_from_other_agents = self._read_observations(OccupancyGridAgent.ObservationDir + "/{}.csv".format(other_rav_name))
#        #update observation manager and also observation file
#        for observation_from_other_agent in observations_from_other_agents:
#            new_observation = self.observation_manager.update_with_observation(observation_from_other_agent)
#            if new_observation:
#                #update observations not already observed
#                self.log_msg_to_cmd("Updating with observation " + str(new_observation) + " from " + other_rav_name, "debug", "cmd_line",self._logged)
#                self.update_observations_file(self.observations_file_loc, new_observation)
#                log_msg_to_file(str(observation_from_other_agent), "info", "observations")
#
#        self.current_belief_map = create_belief_map_from_observations(self.grid, self.agent_name, self.observation_manager.get_all_observations(), self.current_belief_map.get_prior())
#        self.others_coordinated_this_timestep.append(other_rav_name)       
         
    def _move_agent(self, new_location):
        self.previous_pos_intended = self.current_pos_intended
        self.current_pos_intended = new_location
        #due to drift and other possible noise the intended position may not equate to measured position at which
        #sensor reading was taken
        #self.current_pos_measured = new_location
        self.explored_grid_locs.append(new_location)
        
    def _select_action(self):
        return self.move_from_bel_map_callable(self.current_belief_map, self.current_pos_intended, self.explored_grid_locs)
        
    
    def iterate_next_timestep(self, other_agent_positions = []):
        ''' 
        At each discrete timestep, the agent chooses an action to take, executes it and then perceives the environment in which it is operating.
        The agent cannot move to a location which another agent is occupying (or is planning to occupy on the same timestep).
        '''
        #choose an action to perform, sending it to the robot to perform it
        self.actuate()
        #perceive the new state of the environment using the agents sensor(s)
        self.perceive()

        #record the sensor measurement in a file
        self.record_sensor_measurement()
        #update the agents belief
        self.update_belief(self.latest_observation)
        
        #coordinate with other agents if possible
        #in future implement a coordination strategy
        for other_active_agent in self.other_active_agents:
            if self.can_coord_with_other(other_active_agent, self.comms_radius):
                self.coord_with_other(other_active_agent)
                
    def coord_with_other(self, other_active_agent):
        '''Requests observations from other active agents and updates this agents belief with the observations 
        received from other agents'''
        #request observations from other agents
        other_agent_observations = self.request_other_agent_observations(other_active_agent)
        #update current agent belief based on other agents belief
        for other_agent_observation in other_agent_observations:
            self.update_belief(other_agent_observation)

    def find_source(self) -> BeliefMapComponent:
        '''
        Assuming there is 1 or 0 sources present in the available set of grid locations, 
        this method instructs the agent to attempt to locate it.
        This is done by executing the following sequence of abstract actions:
    
            1). Initialize the belief map given the information the agent is initialized with 
            2). Choose an action to explore the region using the strategy provided by move_from_bel_map_callable
            3). Gather sensor data using a sensor model
            4). Update agent belief based on the gathered sensor data
            5). Terminate if termination criteria met (based on agent belief) else continue from 2.
    
        This is not suitable for multi-agent simulations, since agents need to coordinate on each time step.
        Instead, these steps will need to be executed for each agent externally, which should be easy to do with the 
        iterate_next_timestep method.
        '''
        while not self.search_terminator.should_end_search(self.current_belief_map):
            
            if self._end_search_prematurely:
                break
            
            #this actuates, perceives, updates belief map in one step
            self.iterate_next_timestep()
            if self.timestep % 1 == 0:
                print("Timestep: ", self.timestep)
                #self.current_belief_map.save_visualisation("D:/OccupancyGrid/Data/BeliefMapData/BeliefEvolutionImages/img{:03.0f}.png".format(self.timestep))
                
        #return the most likely component of the belief map. This could be the component that represents the source is not present at all
        print(self.current_belief_map.current_belief_vector.get_estimated_state())
        print("source located at {}".format(self.current_belief_map.get_ith_most_likely_component(1).grid_loc))
        #return self.current_belief_map.get_most_likely_component()
        return self.current_belief_map.get_ith_most_likely_component(1)
        

class MultipleSourceDetectingGridAgent(SimpleGridAgent):
    '''
    This agent extends the simple grid agent and is designed to locate multiple sources (or possibly none) of evidence located in 
    a scenario.
    '''
    def __init__(self, grid, initial_pos, move_from_bel_map_callable, height, agent_name, occupancy_sensor_simulator, belief_map_class, init_belief_map, search_terminator, other_active_agents = [], comms_radius = 1000, logged = True):
        super().__init__(grid, initial_pos, move_from_bel_map_callable, height, agent_name, occupancy_sensor_simulator, belief_map_class, init_belief_map, search_terminator, other_active_agents, comms_radius, logged)
        #check that the estimated state can be suitably modified to remove an observed source of evidence
        assert issubclass(self.current_belief_map.current_belief_vector.__class__, BeliefVectorMultipleSources)

    def reset(self):
        pass
    
    def find_sources(self, max_no_sources):
        '''
        Given an upper limit on the number of sources available in the agent's environment, executes a control loop to locate
        the sources, or return a given number of sources as found. This is not suitable for multi-agent simulations, where agents
        should coordinate across each timestep.
        '''
        self.max_no_sources = max_no_sources
        #the locations of sources of evidence that have already been successfully located
        self.located_sources = []
        while len(self.located_sources) < self.max_no_sources:
            next_source = self.find_source()
            #check if the source is deemed to not be present at all
            #if so, break the loop
            #This is bad practice - try and fix in future
            print("next_source: ", next_source)
            if next_source.grid_loc == Vector3r(-1, -1):
                break
            #given the next located source, append it to the list of located sources and then 
            #modify the belief vector to set the probability of subsequent sources to be found at
            #the given location to be zero.
            self.located_sources.append(next_source)
            #
            self.current_belief_map.mark_source_as_located(next_source.grid_loc)
        #return the list of located sources to the user
        return self.located_sources
        
        
class MultipleSourceDetectingGridAgentWithBattery(MultipleSourceDetectingGridAgent):
    '''
    This agent extends the simple grid agent and is designed to locate multiple sources (or possibly none) of evidence located in 
    a scenario. It includes a battery model.
    '''
    def __init__(self, grid, initial_pos, move_from_bel_map_callable, height, agent_name, occupancy_sensor_simulator, battery_capacity_simulator, belief_map_class, init_belief_map, search_terminator, other_active_agents = [], comms_radius = 1000, logged = True, no_battery_levels = 11, charging_locations = None):
        #default number of battery levels is 0-10 = 11 levels
        super().__init__(grid, initial_pos, move_from_bel_map_callable, height, agent_name, occupancy_sensor_simulator, belief_map_class, init_belief_map, search_terminator, other_active_agents, comms_radius, logged)
        #initialize battery model.
        self.battery_estimated_state_model = DefaultStochasticBatteryHMM(no_battery_levels)
        self.battery_capacity_simulator = battery_capacity_simulator
        #use this to terminate simulation if battery drops to 0
        self.__prev_battery_readings = []
        if not isinstance(charging_locations, list):
            self.charging_locations = charging_locations

    def reset(self):
        pass
    
    def terminate_agent(self):
        self.end_comms_server()
        
        
#ToDo: Edit this to coordinate clocks with other agents. Maybe current agent shouldn't move until timestep is
#equal to other agents?
    def _execute_action(self, action):
        '''
        Updates simulators with action performed. Derived agents would request their physical actuators to carry this out.
        '''
        #Before agent executes action, maybe it should coordinate with other agents
        if action[0] == 'move':
            #need to implement this, first have agent coordinate with others to check if they are intending to move to current
            #grid location
#            if action[1] in other_agent_positions: 
#                #this simulates a lower-level collision avoidence agent instructing 
#                #robot to not crash into others
#                return None
            #update previous and current positions if the battery is not critical
            if self.battery_capacity_simulator.get_current_capacity() <= 0.01:
                #this simulates the agents battery failing
                print("BATTERY FAILURE - AGENT WILL NOT MOVE ANY MORE")
                #end the search prematurely since the agent can't move
                self._end_search_prematurely = True
                return None
            else:
                self.previous_pos_intended = self.current_pos_intended
                self.current_pos_intended = action[1]
                
        elif action[0] == 'recharge':
            #request battery simulator to recharge for this timestep
            #could iterate ahead a few timesteps here to simulate recharge time
            self.battery_capacity_simulator.recharge_to_percentage(self.battery_capacity_simulator.get_current_capacity() + 0.1)
            
        else:
            raise Exception("Invalid action requested by {}: {}".format(self.agent_name, action))
      
    def update_battery_belief(self, action):
        '''
        Updates the estimated state ("belief") of the battery capacity
        '''
        if action[0] == 'recharge':
            '''
            Updates battery belief based on percept from simulator given that the battery has recharged
            '''
            if self.current_pos_intended in self.charging_locations:
                self.battery_hmm.update_estimated_state('recharge', self.battery_capacity_simulator.get_current_capacity())
            else:
                print("{} ATTEMPTED TO RECHARGE BATTERY BUT IS NOT AT A RECHARGE LOCATION".format(self.agent_name))
        
        elif action[0] == 'move':
            '''
            Updates battery belief based on percept from simulator given that the agent has moved.
            in reality, would instruct RAV to move to desired location and sample battery capacity percepts at fixed 
            intervals. Updates to the battery_capacity hmm would occur independently as each of these percepts is recorded
            simulated_sensor_readings = []
            '''
            distance_to_travel = self.previous_pos_intended.distance_to(self.current_pos_intended)
            no_seconds = distance_to_travel / self.operational_speed
            no_updates = round(no_seconds)

            
        
            for _ in range(no_updates):
                self.battery_capacity_simulator.move_by_dist_at_speed(self.operational_speed)
                self.__prev_battery_readings.append(self.battery_capacity_simulator.get_current_capacity())
                self.battery_hmm.update_estimated_state('move', round(self.battery_capacity_simulator.get_current_capacity()))
                

    def iterate_next_timestep(self, other_agent_positions = []):
        ''' 
        At each discrete timestep, the agent chooses an action to take, executes it and then perceives the environment in which it is operating.
        The agent cannot move to a location which another agent is occupying (or is planning to occupy on the same timestep).
        '''
        #choose an action to perform, sending it to the robot to perform it
        self.actuate()
        #perceive the new state of the environment using the agents sensor(s)
        self.perceive()
        
        #first update the battery belief state, which is independent of the source locations belief state
        self.update_battery_belief(self.current_pos_intended)

        #record the sensor measurement in a file
        self.record_sensor_measurement()
        #update the agents belief
        self.update_belief(self.latest_observation)
        
        #coordinate with other agents if possible
        #in future implement a coordination strategy
        for other_active_agent in self.other_active_agents:
            if self.can_coord_with_other(other_active_agent, self.comms_radius):
                self.coord_with_other(other_active_agent)




































    
#%%
if __name__ == '__main__':
    pass