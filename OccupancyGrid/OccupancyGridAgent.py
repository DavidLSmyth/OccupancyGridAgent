# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 13:17:07 2018

@author: 13383861
"""

import sys
sys.path.append('..')
import os
import time
import logging
import random

import numpy as np

import AirSimInterface.client as airsim
#from AirSimInterface.types import *

#from AirSimInterface.types import Vector3r

from Utils.Vector3r import Vector3r
from Utils.UE4Grid import UE4Grid
import matplotlib.pyplot as plt
from Utils.ImageAnalysis import sensor_reading
from Utils.AgentObservation import (AgentObservation, AgentObservations, 
                                    _init_observations_file, _update_observations_file,
                                    read_agent_observations_for_analysis_file,
                                    get_agent_observations_file_header,
                                    get_agent_observation_for_csv)

from Utils.ObservationSetManager import ObservationSetManager

from Utils.BeliefMap import (BeliefMap, create_belief_map, 
                             create_belief_map_from_observations, 
                             BeliefMapComponent, calc_posterior, create_single_source_belief_map)

from Utils.AgentAnalysis import (AgentAnalysisState, AgentAnalysisMetadata,
                                 get_agent_state_for_analysis, _get_agent_state_for_analysis,
                                 _init_state_for_analysis_file, _update_state_for_analysis_file,
                                 _init_agent_metadata_file,_write_to_agent_metadata_file,
                                 get_agent_analysis_file_header)

from Utils.Logging import (setup_file_logger,
                           setup_command_line_logger,
                           log_msg_to_file,
                           log_msg_to_cmd)

from Utils.SimpleGrid import SimpleCoord, SimpleGrid

from Utils.Sensors import RadSensor, AirsimImageSensor


logging.basicConfig(filemode = 'w', level = logging.WARNING)
root_logger = logging.getLogger('')
log_directory = "D:/ReinforcementLearning/DetectSourceAgent/Logging/" + time.ctime().replace(' ','').replace(':','_') + "/"
os.mkdir(log_directory)
general_formatter = logging.Formatter("%(asctime)s %(levelname)s:%(name)s:%(funcName)s:%(message)s")
csv_formatter = logging.Formatter("%(message)s")
cmd_line_formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
#stop all messages being propogated up to stdout

#maybe implment this in future
#DEBUG_LEVEL_DATA_NUM = 9 
#logging.addLevelName(DEBUG_LEVEL_DATA_NUM, "DEBUG_LEVEL_DATA")

#create a base agent class

class BaseROCSAFEAgent:
    '''Base class for all agents related to the ROCSAFE project, contains minimal functionality. Designed with goal main in mind
    to be able to compare and measure agent performance in a consistent way'''
    pass
    
class BaseGridAgent:
    
    '''Base class for all agents that use a grid representation of the environment, contains minimal functionality. Designed with goal main in mind
    to be able to compare and measure agent performance in a consistent way'''
    
    ImageDir = 'D:/ReinforcementLearning/DetectSourceAgent/Data/SensorData'
    #stores analysis csvs. Each csv contains agent state at each timestep
    AgentStateDir = "D:/ReinforcementLearning/DetectSourceAgent/Analysis"
    #stores observation json
    ObservationDir = "D:/ReinforcementLearning/DetectSourceAgent/Observations"
    MockedImageDir = 'D:/ReinforcementLearning/DetectSource/Data/MockData'
    
    def __init__(self, grid, initial_pos, move_from_bel_map_callable, height, agent_name, sensor, other_active_agents = [], prior = {}, comms_radius = 1000, logged = True, single_source = False, false_positive_rate=None, false_negative_rate=None):
        
        #list expected types of everything here and check in future
        self.grid = grid
        
        #Record whether want to configure problem for single source or multiple sources
        self.single_source = single_source
        self.false_positive_rate = false_positive_rate
        self.false_negative_rate = false_negative_rate
        
        self._logged = logged
        self._initial_pos = initial_pos
        self.prior = prior
        
        self.current_pos_intended = initial_pos
        self.current_pos_measured = initial_pos
        self.grid_locs = grid.get_grid_points()
        
        #initialise explored grid locations as empty
        self.explored_grid_locs = []
        
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
        self.sensor = sensor
        if self._logged:
            self.setup_logs()

        self.log_msg_to_cmd("Agent " + agent_name + " is alive." , "debug", "cmd_line", self._logged)
        
        #If single source, use modified single source belief map. 
        if single_source:
            print("Using belief map with single source update rule")
            self.current_belief_map = create_single_source_belief_map(self.grid, prior, false_positive_rate, false_negative_rate)
        else:
            print("Using belief map with multiple source update rule (modelling parameter theta at all grid locations independently of each other)")
            self.current_belief_map = create_belief_map(self.grid, prior)

        self.agent_state_file_loc = BaseGridAgent.AgentStateDir + "/{}.csv".format(self.agent_name)
        self.observations_file_loc = BaseGridAgent.ObservationDir + "/{}.csv".format(self.agent_name)
        
        self.init_state_for_analysis_file(self.agent_state_file_loc)
        self.init_observations_file(self.observations_file_loc)
        
        
        
    def reset(self):
        self.__init__(self.grid, self.__initial_pos, self.move_from_bel_map_callable, self.rav_operational_height, self.agent_name, self.sensor, self.other_active_agents, self.current_belief_map.get_prior(), self.comms_radius, self._logged, self.single_source, self.false_positive_rate, self.false_negative_rate)
        
    def log_msg_to_cmd(self, msg, level, log_name, should_log = True):
        if should_log:
            log_msg_to_cmd(msg, level, log_name, should_log)
        
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
    
    def get_agent_state_for_analysis(self):
        return AgentAnalysisState(self.current_pos_intended, self.current_pos_measured,
                                  self.timestep, time.time(), self.get_agent_name(), 
                                  #100% battery capacity, modify this to an object that keeps track of the agents battery
                                  self.total_dist_travelled, 100,
                                  self.prop_battery_cap_used,
                                  self.current_reading, 
                                  self.others_coordinated_this_timestep)
        
    def get_agent_name(self):
        return self.agent_name

    def get_sensor_reading(self):
        #get sensor probability at current position
        #print(self.current_pos_measured)
        return self.sensor.get_probability(self.current_pos_measured)
    
    def actuate(self):
        pass
    
    def perceive(self):
        pass
    
    def get_state(self):
        pass
    
    def move_agent(self):
        self.timestep+=1

        
class SimpleGridAgent(BaseGridAgent):
    def __init__(self, grid, initial_pos, move_from_bel_map_callable, height, agent_name, sensor, other_active_agents = [], prior = {}, comms_radius = 1000, logged = True, single_source = False, false_positive_rate=None, false_negative_rate=None):
        super().__init__(grid, initial_pos, move_from_bel_map_callable, height, agent_name, sensor, other_active_agents, prior, comms_radius, logged, single_source, false_positive_rate, false_negative_rate)
        #for simple grid agent assume it is in correct grid square
        self.current_pos_measured = initial_pos
        self.no_greedy_moves = 0
        
    def reset(self):
        self.__init__(self.grid, self._initial_pos, self.move_from_bel_map_callable, self.rav_operational_height, self.agent_name, self.sensor, self.other_active_agents, self.current_belief_map.get_prior(), self.comms_radius, self._logged, self.single_source, self.false_positive_rate, self.false_negative_rate)

        
    def move_agent(self, location: SimpleCoord, other_agent_positions):
        #if self.current_pos_intended not in other_agent_positions:
        #for now let agents occupy the same grid locations
        super().move_agent()
        self.current_pos_intended = location
            #due to drift and other possible noise the intended position may not equate to measured position at which
            #sensor reading was taken
        self.current_pos_measured = location
        self.explored_grid_locs.append(location)
        #else:
            #don't do anything if agent will crash into other agent
         #   pass
        
        

    def coord_with_other(self, other_rav_name):
        '''coordinate with other rav by requesting their measurement list and sending our own measurement list first write own measurement list to file'''
        log_msg_to_file(str(other_rav_name) + ' ' + str(self.timestep), "info", "comms", self._logged)
        self.log_msg_to_cmd("Coordinating with: " + other_rav_name, "debug", "cmd_line", self._logged)
        observations_from_other_agents = self._read_observations(OccupancyGridAgent.ObservationDir + "/{}.csv".format(other_rav_name))
        #update observation manager and also observation file
        for observation_from_other_agent in observations_from_other_agents:
            new_observation = self.observation_manager.update_with_observation(observation_from_other_agent)
            if new_observation:
                #update observations not already observed
                self.log_msg_to_cmd("Updating with observation " + str(new_observation) + " from " + other_rav_name, "debug", "cmd_line",self._logged)
                self.update_observations_file(self.observations_file_loc, new_observation)
                log_msg_to_file(str(observation_from_other_agent), "info", "observations")

        self.current_belief_map = create_belief_map_from_observations(self.grid, self.agent_name, self.observation_manager.get_all_observations(), self.current_belief_map.get_prior())
        self.others_coordinated_this_timestep.append(other_rav_name)       
    
    def can_coord_with_other(self, other_rav_name, range_m):
        #check if any other ravs in comm radius. if so, return which ravs can be communicated with
        #assume communcications randomly drop with probability in proportion to range_m
        #for now 10% communication
        return random.random() < 0.1
        #return self.rav.can_coord_with_other_stochastic(self.get_agent_name(), other_rav_name, range_m)
    
    def explore_timestep(self, other_agent_positions):
        '''Gets rav to explore next timestep'''
        #belief_map: BeliefMap, current_grid_loc: Vector3r, explored_grid_locs: 'list of Vector3r'
        greedy_move, next_pos = self.move_from_bel_map_callable(self.current_belief_map, self.current_pos_intended, [])
        if greedy_move:
            self.no_greedy_moves += 1
        
        self.move_agent(next_pos, other_agent_positions)      
        #record image at location
        #self.record_image()
        #get sensor reading, can be done on separate thread        
        #self.current_reading
        self.current_reading = self.get_sensor_reading()
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



#this is the agent to use for testing with rad source
class SimpleGridAgentWithSources(SimpleGridAgent):
    
    def __init__(self, grid, initial_pos, move_from_bel_map_callable, height, agent_name, sensor, other_active_agents = [], prior = {}, comms_radius = 1000, logged = True, single_source = False, false_positive_rate=None, false_negative_rate=None):
        super().__init__(grid, initial_pos, move_from_bel_map_callable, height, agent_name, sensor, other_active_agents, prior, comms_radius, logged, single_source, false_positive_rate, false_negative_rate)
        
    def reset(self):
        self.__init__(self.grid, self._initial_pos, self.move_from_bel_map_callable, self.rav_operational_height, self.agent_name, self.sensor, self.other_active_agents, self.current_belief_map.get_prior(), self.comms_radius, self._logged, self.single_source, self.false_positive_rate, self.false_negative_rate)
    
    
    def explore_timestep(self, other_agent_positions):
        '''Gets rav to explore next timestep'''
        greedy_move, next_pos = self.move_from_bel_map_callable(self.current_belief_map, self.current_pos_intended, self.explored_grid_locs)
        
        if greedy_move:
            self.no_greedy_moves += 1
            
        self.move_agent(next_pos, other_agent_positions)
        #get sensor reading
        self.current_reading = self.get_sensor_reading()
        
        #update belief map based on sensor readings - maybe it makes more sense to update based on the agent observation
        self.current_belief_map.update_from_prob(self.current_pos_intended, self.current_reading)
        #create to observation object with timestamp etc.
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


class OccupancyGridAgent():
    '''agent that moves around an occupancy grid in order to locate a source of radiation. Uses a rav agent'''
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
    
if __name__ == "__main__":
    from Utils.ClientMock import KinematicsState, MockRavForTesting, ImageType, Vector3r
    grid = UE4Grid(15, 20, Vector3r(0,0), 60, 45)
    #grid, move_from_bel_map_callable, height, epsilon, multirotor_client, agent_name, performance_csv_path: "file path that agent can write performance to", prior = []
    #grid, initial_pos, move_from_bel_map_callable, height, epsilon, multirotor_client, agent_name, prior = {}
    occupancy_grid_agent = OccupancyGridAgent(grid, Vector3r(0,0), get_move_from_belief_map_epsilon_greedy, -12, 0.2, MockRavForTesting(), 'agent1')
    #write some tests for agent here
    occupancy_grid_agent.current_pos_intended = Vector3r(0,0)
    occupancy_grid_agent.current_pos_measured = None
    occupancy_grid_agent.current_reading = 0.1
    occupancy_grid_agent.get_agent_state_for_analysis()
    occupancy_grid_agent.explore_timestep()
        
    #belief_map: BeliefMap, current_grid_loc: Vector3r, epsilon: float, eff_radius = None) -> Vector3r:
    dont_move = lambda belief_map, current_grid_loc, epsilon:  Vector3r(15,20)
    print(grid.get_grid_points())
    ###Check that agents can communicate with each other
    occupancy_grid_agent1 = OccupancyGridAgent(grid, Vector3r(0,0), get_move_from_belief_map_epsilon_greedy, -12, 0.2, MockRavForTesting(), 'agent1', ['agent2'])
    occupancy_grid_agent2 = OccupancyGridAgent(grid, Vector3r(15,20), dont_move, -12, 0.2, MockRavForTesting(), 'agent2', ['agent1'])
    
    print(occupancy_grid_agent2.can_coord_with_other(occupancy_grid_agent1.agent_name, 10))
    occupancy_grid_agent1.explore_timestep()
    occupancy_grid_agent2.explore_timestep()
    occupancy_grid_agent1.coord_with_other("agent2")
    assert occupancy_grid_agent1.current_belief_map.get_belief_map_component(Vector3r(15,20)).likelihood < 0.1
    #assert occupancy_grid_agent1.current_belief_map.get_belief_map_component(Vector3r(20,15)).likelihood < 0.1

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    