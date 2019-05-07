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
import configparser
import pickle
import subprocess
import typing
import random

from abc import ABC, abstractmethod
import numpy as np

from Utils.Vector3r import Vector3r
from Utils.UE4Grid import UE4Grid

from Utils.AgentObservation import (_init_observations_file, _update_observations_file,
                                    get_agent_observations_file_header, AgentObservation)

from Utils.ObservationSetManager import ObservationSetManager



from Utils.AgentAnalysis import (AgentAnalysisState, AgentAnalysisMetadata,
                                 get_agent_state_for_analysis, _get_agent_state_for_analysis,
                                 _init_state_for_analysis_file, _update_state_for_analysis_file,
                                 _init_agent_metadata_file,_write_to_agent_metadata_file,
                                 get_agent_analysis_file_header)

from Utils.Logging import (setup_file_logger,
                           setup_command_line_logger,
                           log_msg_to_file,
                           log_msg_to_cmd)

from Communication.CommsClient import AgentCommunicatorClient
from Communication.CommsServer import AgentCommunicatorServer
#%%
logging.basicConfig(filemode = 'w', level = logging.WARNING)
root_logger = logging.getLogger('')
log_directory = "D:/ReinforcementLearning/DetectSourceAgent/Logging/" + time.ctime().replace(' ','').replace(':','_') + "/"
os.mkdir(log_directory)
general_formatter = logging.Formatter("%(asctime)s %(levelname)s:%(name)s:%(funcName)s:%(message)s")
csv_formatter = logging.Formatter("%(message)s")
cmd_line_formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
#stop all messages being propogated up to stdout

#DEBUG_LEVEL_DATA_NUM = 9 
#logging.addLevelName(DEBUG_LEVEL_DATA_NUM, "DEBUG_LEVEL_DATA")


class BaseGridAgent:
    
    '''
    Base class for all agents that use a grid representation of the environment, 
    contains minimal functionality. Designed with goal main in mind
    to be able to compare and measure agent performance in a consistent way
    '''
    
    #ImageDir = 'D:/ReinforcementLearning/DetectSourceAgent/Data/SensorData'
    #stores analysis csvs. Each csv contains agent state at each timestep
    #AgentStateDir = "D:/ReinforcementLearning/DetectSourceAgent/Analysis"
    #stores observation json
    #ObservationDir = "D:/ReinforcementLearning/DetectSourceAgent/Observations"
    #MockedImageDir = 'D:/ReinforcementLearning/DetectSource/Data/MockData'
    
    def __init__(self, grid, initial_pos, move_from_bel_map_callable, height, agent_name, occupancy_sensor_simulator, belief_map_class, init_belief_map, other_active_agents = [], initial_estimated_state = np.array([]), comms_radius = 1000, logged = True):
        
        '''
        occupancy_simluator returns a simulated reading of whether or not a grid cell is occupied (by a source of evidence)
        '''
        
        #configures the directory conventions for storing data
        self.configure_file_conventions()
        #list expected types of everything here and check in future
        self.grid = grid
        
        #Record whether want to configure problem for single source or multiple sources
#        self.false_positive_rate = false_positive_rate
#        self.false_negative_rate = false_negative_rate
        
        self._logged = logged
        self._initial_pos = initial_pos
        self.prior = initial_estimated_state
        
        #intended position (last control action)
        self.current_pos_intended = initial_pos
        #measured position
        self.current_pos_measured = initial_pos
        
        #valid grid locations to explore
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
        
        #intialise the communications variables
        self.comms_radius = comms_radius
        self.start_comms_server()
        self.comms_client = AgentCommunicatorClient()
        self.others_coordinated_this_timestep = []
        
        #manages observations of this agent and other agents
        self.observation_manager = ObservationSetManager(self.agent_name)
        
        #maybe should include possibility of multiple sensors?
        self.occupancy_sensor_simulator = occupancy_sensor_simulator
        if self._logged:
            self.setup_logs()

        self.log_msg_to_cmd("Agent " + agent_name + " is alive." , "debug", "cmd_line", self._logged)
        
#        #If single source, use modified single source belief map. 
#        if single_source:
#            print("Using belief map with single source update rule")
#            self.current_belief_map = create_single_source_belief_map(self.grid, prior, false_positive_rate, false_negative_rate)
#        else:
#            print("Using belief map with multiple source update rule (modelling parameter theta at all grid locations independently of each other)")
#            self.current_belief_map = create_belief_map(self.grid, prior)
        
        #initialise the current belief map with the prior provided by the user
        #belief_map_class is a class of belief map that should obey an interface that allows recursive filtering
        self.current_belief_map = init_belief_map


        self.agent_state_file_loc = self.directories_mapping['AgentStateDir'] + "/{}.csv".format(self.agent_name)
        self.observations_file_loc = self.directories_mapping['ObservationDir'] + "/{}.csv".format(self.agent_name)
        
        #self.init_state_for_analysis_file(self.agent_state_file_loc)
        self.init_observations_file(self.observations_file_loc)
        
    def configure_file_conventions(self):
        '''Configures which directory files should be saved'''
        config_loc = './OccupancyGrid/Agents/BaseOccupancyGridAgentconfig.ini'
        config_parser = configparser.ConfigParser()
        config_parser.read(config_loc)
        self.directories_mapping = dict(config_parser['DIRECTORIES'])
    
    
    def pickle_to_file(self, file_loc):
        with open(file_loc, 'wb') as f:
            pickle.dump(self, f)
        
    def start_comms_server(self):
        subprocess.run(["python", "./Communication/CommsServer.py", self.agent_name])
        
    def end_comms_server(self):
        self.comms_client.shutdown_server(self.agent_name)
        
    def reset(self):
        '''Resets the agent to its initial state'''
        self.battery.reset()
        self.__init__(self.grid, self.__initial_pos, self.move_from_bel_map_callable, self.rav_operational_height, self.agent_name, self.sensor, self.belief_map_class, self.other_active_agents, self.current_belief_map.get_prior(), self.comms_radius, self._logged)
        
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

    @abstractmethod
    def _get_sensor_readings(self) -> typing.List[AgentObservation]:
        '''Returns the readings of the agents sensors at it's current location. Sensor readings 
        are currently returned as agent observations, which describe the reading, location, timestep, 
        timestamp, location.'''
        #get sensor probability at current position
        #print(self.current_pos_measured)
        #return self.sensor.get_probability(self.current_pos_measured)
        pass
    
    def actuate(self):
        '''
        Select an action to take and then execute that action in the environment. This is assumed to be deterministic.
        '''
        self.increment_timestep()
        #actions are assumed to tell agent where to move
        action = self._select_action()
        self.current_pos_intended = action
        
        
        #for now action can be to move the agent to a grid location, or to recharge the battery
        #action = self._select_action()
        #if isinstance(action, Vector3r):
        #    self._move_agent(action)
#        otherwise if the action is to recharge the agent's battery, do so.
#        elif isinstance(action, ):
#            maybe should give the agent the option to navigate to recharge point and then recharge?
#            maybe another option should be to keep recharging at the current timestep?             
#            self.battery.recharge_to_percentage()
#        #attempts to recharge the battery to a percentage
        
        
    def perceive(self):
        '''
        Use on-board sensors to percieve the environment and store the perceptions for use in future computations regarding
        which actions to take
        '''
        self.current_sensor_reading = self.occupancy_sensor_simulator.get_reading(self.current_pos_intended)
        
    def record_sensor_measurement(self):
        '''
        Records the most recently percieved sensor measurement in the observations file.
        '''
        self.latest_observation = self.AgentObservation(self.current_pos_intended, self.current_sensor_reading, self.timestep, time.time(), self.agent_name)
        self.update_observations_file(self.observations_file_loc, latest_observation)
        log_msg_to_file(get_agent_observation_for_csv(latest_observation), "info", "observations")
        
    def update_belief(self, observation):
        '''Updates the agents belief based on a sensor reading'''
        self.current_belief_map.update_from_observation(observation)
    
    @abstractmethod
    def get_belief(self):
        '''
        Gets the belief of the agent at the current time step. This is a distribution over all possible states given observations.
        '''
        pass
    
    @abstractmethod
    def _move_agent(self, new_location):
        '''
        Responsible for actually sending commands to actuators to move the agent.
        '''
        pass
    
    @abstractmethod
    def _select_action(self):
        '''
        A method that represent agent 'deliberation' over which action should be executed at the next timestep. Actions for now
        are simply assumed to be a new grid location to move to, and whether to recharge the battery or not.
        '''
        pass
        
    def increment_timestep(self):
        self.timestep+=1
        
    def request_other_agent_observations(self, other_agent_name):
        '''
        Requests other agent to send all observations that it has gathered and been sent up to the current point in time
        '''
        try:
            return self.comms_client.get_observations_from(other_agent_name)
        except Exception as e:
            #if there are comms problems, return an empty list and log the problem to stdout for now
            print("Could not communicate with agent {}".format(other_agent_name))
            print(e)
            return []
        
    
    #def request_belief_map_from_other(self, other_rav_name):
    #    try:
            #attempts to send the belief distribution over all possible states given sensor readings to date
    #        return self.comms_client.send_belief(self.current_belief_map.)
    
    def can_coord_with_other(self, other_rav_name, range_m):
        '''
        Defines the probability with which the current agent can successfully 
        coordinate with other agent as a function of distance
        '''
        #check if any other ravs in comm radius. if so, return which ravs can be communicated with
        #assume communcications randomly drop with probability in proportion to range_m
        #for now 10% communication. This should probably go in a config file
        return random.random() < 0.1

#%%
    
if __name__ == "__main__":
    from Utils.ClientMock import KinematicsState, MockRavForTesting, ImageType, Vector3r
    grid = UE4Grid(15, 20, Vector3r(0,0), 60, 45)
    #grid, move_from_bel_map_callable, height, epsilon, multirotor_client, agent_name, performance_csv_path: "file path that agent can write performance to", prior = []
    #grid, initial_pos, move_from_bel_map_callable, height, epsilon, multirotor_client, agent_name, prior = {}
    occupancy_grid_agent = BaseGridAgent(grid, Vector3r(0,0), get_move_from_belief_map_epsilon_greedy, -12, 0.2, MockRavForTesting(), 'agent1')
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
    occupancy_grid_agent1 = BaseGridAgent(grid, Vector3r(0,0), get_move_from_belief_map_epsilon_greedy, -12, 0.2, MockRavForTesting(), 'agent1', ['agent2'])
    occupancy_grid_agent2 = BaseGridAgent(grid, Vector3r(15,20), dont_move, -12, 0.2, MockRavForTesting(), 'agent2', ['agent1'])
    
    print(occupancy_grid_agent2.can_coord_with_other(occupancy_grid_agent1.agent_name, 10))
    occupancy_grid_agent1.explore_timestep()
    occupancy_grid_agent2.explore_timestep()
    occupancy_grid_agent1.coord_with_other("agent2")
    assert occupancy_grid_agent1.current_belief_map.get_belief_map_component(Vector3r(15,20)).likelihood < 0.1
    #assert occupancy_grid_agent1.current_belief_map.get_belief_map_component(Vector3r(20,15)).likelihood < 0.1

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    