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

from Utils.Vector3r import Vector3r
from Utils.UE4Grid import UE4Grid
import subprocess
from Utils.AgentObservation import (_init_observations_file, _update_observations_file,
                                    read_agent_observations_for_analysis_file,
                                    get_agent_observations_file_header)

from Utils.ObservationSetManager import ObservationSetManager

from Utils.BeliefMap import (create_belief_map, create_single_source_belief_map)

from Utils.AgentAnalysis import (AgentAnalysisState, AgentAnalysisMetadata,
                                 get_agent_state_for_analysis, _get_agent_state_for_analysis,
                                 _init_state_for_analysis_file, _update_state_for_analysis_file,
                                 _init_agent_metadata_file,_write_to_agent_metadata_file,
                                 get_agent_analysis_file_header)

from Utils.Logging import (setup_file_logger,
                           setup_command_line_logger,
                           log_msg_to_file,
                           log_msg_to_cmd)

from Communication.CommsAgent import AgentCommunicatorClient, AgentCommunicatorServer

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
    
    '''Base class for all agents that use a grid representation of the environment, contains minimal functionality. Designed with goal main in mind
    to be able to compare and measure agent performance in a consistent way'''
    
    #ImageDir = 'D:/ReinforcementLearning/DetectSourceAgent/Data/SensorData'
    #stores analysis csvs. Each csv contains agent state at each timestep
    #AgentStateDir = "D:/ReinforcementLearning/DetectSourceAgent/Analysis"
    #stores observation json
    #ObservationDir = "D:/ReinforcementLearning/DetectSourceAgent/Observations"
    #MockedImageDir = 'D:/ReinforcementLearning/DetectSource/Data/MockData'
    
    def __init__(self, grid, initial_pos, move_from_bel_map_callable, height, agent_name, sensor, other_active_agents = [], prior = {}, comms_radius = 1000, logged = True, single_source = False, false_positive_rate=None, false_negative_rate=None):
        
        #configures the directory conventions for storing data
        self.configure_file_conventions()
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
        self.start_comms_server()
        self.comms_client = AgentCommunicatorClient()
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

        self.agent_state_file_loc = self.directories_mapping['AgentStateDir'] + "/{}.csv".format(self.agent_name)
        self.observations_file_loc = self.directories_mapping['ObservationDir'] + "/{}.csv".format(self.agent_name)
        
        self.init_state_for_analysis_file(self.agent_state_file_loc)
        self.init_observations_file(self.observations_file_loc)
        
    def configure_file_conventions(self):
        config_loc = './OccupancyGrid/Agents/BaseOccupancyGridAgentconfig.ini'
        config_parser = configparser.ConfigParser()
        config_parser.read(config_loc)
        self.directories_mapping = dict(config_parser['DIRECTORIES'])
        
    def init_belief_map(self, single_source):
        pass
        
    def start_comms_server(self):
        subprocess.run(["python", "./Communication/CommsServer.py", self.agent_name])
        
    def end_comms_server(self):
        self.comms_client.shutdown_server(self.agent_name)
        
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

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    