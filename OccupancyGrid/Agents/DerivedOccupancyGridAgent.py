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
        
    def move_agent(self, location: Vector3r, other_agent_positions):
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
        self.current_pos_intended = new_location
        #due to drift and other possible noise the intended position may not equate to measured position at which
        #sensor reading was taken
        self.current_pos_measured = new_location
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
            #this actuates, perceives, updates belief map in one step
            self.iterate_next_timestep()
            if self.timestep % 1 == 0:
                print("Timestep: ", self.timestep)
                self.current_belief_map.save_visualisation("D:/OccupancyGrid/Data/BeliefMapData/BeliefEvolutionImages/img{:03.0f}.png".format(self.timestep))
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
        if not isinstance(charging_locations, list):
            self.charging_locations = charging_locations

    def reset(self):
        pass
    
    def move_agent(self, location: Vector3r, other_agent_positions):
        if location in other_agent_positions: 
            #this simulates a lower-level collision avoidence agent instructing 
            #robot to not crash into others
            return None
        elif self.battery_capacity_simulator.get_current_capacity <= 0.01:
            #this simulates the agents battery failing
            print("BATTERY FAILURE - AGENT WILL NOT MOVE ANY MORE")
            return None
        else:
            self._move_agent(location)
      
    def update_battery_belief(self, new_location):
        
        self.current_pos_intended = new_location
        #due to drift and other possible noise the intended position may not equate to measured position at which
        #sensor reading was taken
        self.current_pos_measured = new_location
        self.explored_grid_locs.append(new_location)
        
        if new_location == self.current_location and new_location in self.charging_locations:
            #assume that agent has decided to recharge for a single timestep
            #this is not consistent with the move actions, for which a variable number of timesteps could be used to update
            #need to get reading from simulated battery once action has been performed
            #this is very messy but for now assume the battery recharge adds 10% of max. capacity per unit timestep
            self.battery_capacity_simulator.self.current_capacity += 0.1
            self.battery_hmm.update_estimated_state('recharge', self.battery_capacity_simulator.get_current_capacity())
            
        else:
            distance_to_travel = new_location.distance_to(self.current_pos_intended)
            no_seconds = distance_to_travel / self.operational_speed
            no_updates = round(no_seconds)
            #in reality, would instruct RAV to move to desired location and sample battery capacity percepts at fixed 
            #intervals. Updates to the battery_capacity hmm would occur independently as each of these percepts is recorded
            #simulated_sensor_readings = []
            for _ in range(no_updates):
                self.battery_capacity_simulator.move_by_dist_at_speed(self.operational_speed)
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
        self.update_battery_belief()

        #record the sensor measurement in a file
        self.record_sensor_measurement()
        #update the agents belief
        self.update_belief(self.latest_observation)
        
        #coordinate with other agents if possible
        #in future implement a coordination strategy
        for other_active_agent in self.other_active_agents:
            if self.can_coord_with_other(other_active_agent, self.comms_radius):
                self.coord_with_other(other_active_agent)



































class OccupancyGridAgent():
    
    '''Agent that moves around an occupancy grid in order to locate a source of radiation. Uses a rav agent'''
    ImageDir = 'D:/ReinforcementLearning/DetectSourceAgent/Data/SensorData'
    #stores analysis csvs. Each csv contains agent state at each timestep
    AgentStateDir = "D:/ReinforcementLearning/DetectSourceAgent/Analysis"
    #stores observation json
    ObservationDir = "D:/ReinforcementLearning/DetectSourceAgent/Observations"
    MockedImageDir = 'D:/ReinforcementLearning/DetectSource/Data/MockData'
    #break apart this into components, one which manages actuation/sensing, one which manages/represents state, etc.
    def __init__(self, grid, initial_pos, move_from_bel_map_callable, height, multirotor_client, agent_name, other_active_agents = [], prior = {}, comms_radius = 100000):
        #list expected types of everything here  
        self.rav = multirotor_client
        self.grid = grid
        self.current_pos_intended = initial_pos
        self.grid_locs = grid.get_grid_points()
        self.timestep = 0
        self.rav_operational_height = height
        self.move_from_bel_map_callable = move_from_bel_map_callable
        self.agent_name = agent_name
        self.agent_states_for_analysis = []
        self.other_active_agents = other_active_agents
        self.total_dist_travelled = 0
        self.distance_covered_this_timestep = 0
        self.prop_battery_cap_used = 0
        self.current_battery_cap = 1
        self.comms_radius = comms_radius
        self.others_coordinated_this_timestep = []
        #manages observations of this agent and other agents
        self.observation_manager = ObservationSetManager(self.agent_name)
        
        self.setup_logs()
        self.log_msg_to_cmd("Agent " + agent_name + " is alive." , "debug", "cmd_line", self._logged)

        self.agent_state_file_loc = OccupancyGridAgent.AgentStateDir + f"/{self.agent_name}.csv"
        self.observations_file_loc = OccupancyGridAgent.ObservationDir + f"/{self.agent_name}.csv"
        
        self.current_belief_map = create_belief_map(self.grid, self.agent_name, prior)
        self.init_state_for_analysis_file(self.agent_state_file_loc)
        self.init_observations_file(self.observations_file_loc)  
        
        
    def setup_logs(self):
        '''Sets up logs so that relevant information can be sent to csvs and other log steams'''       
        setup_file_logger("move",log_directory+self.agent_name+"move_log.log", general_formatter)
        setup_file_logger("state",log_directory+self.agent_name+"state_log.log",csv_formatter)
        setup_file_logger("comms",log_directory+self.agent_name+"comms_log.log", general_formatter)
        setup_file_logger("observations",log_directory+self.agent_name+"observations_log.log", csv_formatter)
        setup_command_line_logger("cmd_line", cmd_line_formatter)
        
        log_msg_to_file(get_agent_analysis_file_header(), "info", "state")
        log_msg_to_file(get_agent_observations_file_header(), "info", "observations")
        
                       
    def __eq__(self, other):
        '''This agent is the same as another agent if names are the same. Refine this later'''
        return self.agent_name == other.agent_name
        
    def get_available_actions(self, state):
        '''Returns actions available to RAV based on its current state'''
        #currently just move, eventually: go_to_charge, keep_charging, stop_charging, etc.
        pass
        
    def get_belief_map_after_t_timesteps(self, t):
        '''Calculates what the agent's belief map would be after t timesteps'''
        pass
    
    def charge_battery(self):
        #request from api if battery can be charged. While charging, capacity goes up. Charging can be cancelled at any point to explore some more.
        pass
        
    def move_agent(self, destination_intended: Vector3r):
        self.rav.moveToPositionAsync(destination_intended.x_val, destination_intended.y_val, self.rav_operational_height, 3, vehicle_name = self.agent_name).join()
        log_msg_to_file(str(destination_intended) + ' ' + str(self.timestep), "info", "move")
        self.log_msg_to_cmd("Moving to destination: " + str(destination_intended) + ' ' + str(self.timestep), "debug", "cmd_line", self._logged)
        self.distance_covered_this_timestep = self.current_pos_intended.distance_to(destination_intended)
        self.total_dist_travelled += self.current_pos_intended.distance_to(destination_intended)
        
        self.prop_battery_cap_used += self.current_battery_cap - self.rav.getRemainingBatteryCap()
        self.current_battery_cap = self.rav.getRemainingBatteryCap()
        self.log_msg_to_cmd("Remaining battery cap: " + str(self.current_battery_cap), "debug", "cmd_line", self._logged)
        
    def get_agent_name(self):
        return self.agent_name
    
    def get_agent(self):
        return self.rav
        
# =============================================================================
#     'AgentAnalysisState', [('position_intended',Vector3r),
#                                                      ('position_measured',Vector3r),
#                                                      ('timestep', int),
#                                                      ('timestamp', float),
#                                                      ('rav_name',str),
#                                                      #maybe add distance travelled for current timestep
#                                                      ('total_dist_travelled', float),
#                                                      ('remaining_batt_cap', float),
#                                                      ('prop_battery_cap_used', float),
#                                                      ('sensor_reading', float),
#                                                      #is it necessary to record the grid along with the likelihoods in case want the grid to 
#                                                      #dynamically change? For now assume grid is fixed and in 1-1 correspondance with likelihoods
#                                                      #'occ_grid_likelihoods',
#                                                      #which other agents did the agent coordinate with on this timestep
#                                                      ('coordinated_with_other_names', list)]
# =============================================================================
    def get_agent_state_for_analysis(self):
        return AgentAnalysisState(self.current_pos_intended, self.current_pos_measured,
                                  self.timestep, time.time(), self.get_agent_name(), 
                                  self.total_dist_travelled, self.rav.getRemainingBatteryCap(),
                                  self.prop_battery_cap_used,
                                  self.current_reading, 
                                  self.others_coordinated_this_timestep)
        
    def init_state_for_analysis_file(self, file_path):
        _init_state_for_analysis_file(file_path)
            
    def update_state_for_analysis_file(self, file_path, agent_state):
        _update_state_for_analysis_file(file_path, agent_state)
        
    def init_observations_file(self, file_path):
        _init_observations_file(file_path)
        
    def update_observations_file(self, file_path, agent_observation):
        _update_observations_file(file_path, agent_observation)
        
    def _read_observations(self, file_loc):
        return read_agent_observations_for_analysis_file(file_loc)
        
        
    #coordination strategy:
    #agent will write all measurements in its possession to a file at each timestep. When communication requested,
    #other agent will read all measurements from the file.
        
    def can_coord_with_other(self, other_rav_name, range_m):
        #check if any other ravs in comm radius. if so, return which ravs can be communicated with
        #assume communcications randomly drop with probability in proportion to range_m
        return self.rav.can_coord_with_other_stochastic(self.get_agent_name(), other_rav_name, range_m)
    
    def coord_with_other(self, other_rav_name):
        '''coordinate with other rav by requesting their measurement list and sending our own measurement list first write own measurement list to file'''
        log_msg_to_file(str(other_rav_name) + ' ' + str(self.timestep), "info", "comms")
        self.log_msg_to_cmd("Coordinating with: " + other_rav_name, "debug", "cmd_line", self._logged)
        observations_from_other_agents = self._read_observations(OccupancyGridAgent.ObservationDir + f"/{other_rav_name}.csv")
        #update observation manager and also observation file
        for observation_from_other_agent in observations_from_other_agents:
            new_observation = self.observation_manager.update_with_observation(observation_from_other_agent)
            if new_observation:
                #update observations not already observed
                self.log_msg_to_cmd("Updating with observation " + str(new_observation) + " from " + other_rav_name, "debug", "cmd_line", self._logged)
                self.update_observations_file(self.observations_file_loc, new_observation)
                log_msg_to_file(str(observation_from_other_agent), "info", "observations")

        self.current_belief_map = create_belief_map_from_observations(self.grid, self.agent_name, self.current_belief_map.get_prior(), self.observation_manager.get_all_observations())
        self.others_coordinated_this_timestep.append(other_rav_name)       
    
    def record_image(self):
        responses = self.rav.simGetImages([ImageRequest("3", ImageType.Scene)], vehicle_name = self.agent_name)
        response = responses.pop()
        # get numpy array
        filename = OccupancyGridAgent.ImageDir + "/photo_" + str(self.timestep)
        airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8) 

    def update_agent_pos_measured(self):
        self.current_pos_measured = self.rav.getMultirotorState().kinematics_estimated.position
    
    def explore_timestep(self):
        '''Gets rav to explore next timestep'''
        
        next_pos = self.move_from_bel_map_callable(self.current_belief_map, self.current_pos_intended, [])
        self.move_agent(next_pos)      
        self.current_pos_intended = next_pos
        self.current_pos_measured = self.rav.getMultirotorState(vehicle_name = self.agent_name).kinematics_estimated.position
        self.update_agent_pos_measured()
        #record image at location
        self.record_image()
        #get sensor reading, can be done on separate thread        
        self.current_reading = float(sensor_reading(OccupancyGridAgent.ImageDir + "/photo_" + str(self.timestep) + '.png')[0])
        self.current_belief_map.update_from_prob(self.current_pos_intended, self.current_reading)
        newest_observation = AgentObservation(self.current_pos_intended, self.current_reading, self.timestep, time.time(), self.agent_name)
        self.log_msg_to_cmd("Newest observation: " + str(newest_observation), "debug", "cmd_line", self._logged)
        self.observation_manager.update_rav_obs_set(self.agent_name, [AgentObservation(self.current_pos_intended, self.current_reading, self.timestep, time.time(), self.agent_name)])
        
        
          #self._write_observations(self.observations_file_loc)
        self.update_state_for_analysis_file(self.agent_state_file_loc, self.get_agent_state_for_analysis())
        log_msg_to_file(self.get_agent_state_for_analysis(), "info", "state")
        self.update_observations_file(self.observations_file_loc, newest_observation)
        log_msg_to_file(get_agent_observation_for_csv(newest_observation), "info", "observations")
        
        #coordinate with other agents if possible
        for other_active_agent in self.other_active_agents:
            if self.can_coord_with_other(other_active_agent, self.comms_radius):
                self.coord_with_other(other_active_agent)

      
        #if agent is in range, communicate
        
    def explore_t_timesteps(self, t: int):
        for i in range(t):
            self.explore_timestep()
            self.timestep += 1
        #print("current belief map: {}".format(self.current_belief_map))
        return self.current_belief_map
    
    
#%%
if __name__ == '__main__':
    pass