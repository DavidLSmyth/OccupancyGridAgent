# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 11:36:33 2019

@author: 13383861
"""

import configparser
import re

from Utils.Vector3r import Vector3r
from Utils.SensorSimulators import FalsePosFalseNegBinarySensorSimulator, BinarySensorParameters
from Utils.UE4Grid import UE4Grid

#%%
def get_agent_config(agent_number):
    '''
    Returns the config file which holds parameters related to the
    agent.
    '''
    parser = configparser.ConfigParser()
    parser.read("./Config/AgentConfigs/agent{}Config.ini".format(agent_number))
    return parser


def get_env_config():
    '''
    Returns the config file which holds parameters related to the
    environment.
    '''
    parser = configparser.ConfigParser()
    parser.read("./Config/EnvConfig.ini")
    return parser


def parse_Vector3r_from_string(string):
    '''
    Given a string with Vector3rs, returns the Vector3r objects in a list in the order that they appeared in the string
    '''
    return [Vector3r(int(_.split(',')[0]), int(_.split(',')[1])) for _ in re.findall("Vector3r\(([0-9]+ ?, ?[0-9]+)\)", string)]

def get_simulated_sensor_from_config(agent_number):
    '''
    Returns a simulated sensor from the config files which is specific to a 
    given agent
    '''
    agent_parser = get_agent_config(agent_number)
    #binary_sensor_parameters: BinarySensorParameters, source_locations: typing.List[Vector3r]
    return FalsePosFalseNegBinarySensorSimulator(BinarySensorParameters(float(agent_parser['SENSOR_SIMULATOR_PARAMETERS']['FalsePositiveRate']),
                                                 float(agent_parser['SENSOR_SIMULATOR_PARAMETERS']['FalseNegativeRate'])), 
                                                 get_source_locations_from_config())
    
                                                 
def get_grid_from_config():
    env_parser = get_env_config()
    grid_params = env_parser['GridParams']
    return UE4Grid(int(grid_params['SpacingX']), int(grid_params['SpacingY']), parse_Vector3r_from_string(grid_params["Origin"])[0],
                   x_lim = int(grid_params['XLim']), y_lim = int(grid_params['YLim']))
    
def get_agent_start_pos_from_config(agent_number):
    agent_parser = get_agent_config(agent_number)
    return parse_Vector3r_from_string(agent_parser["InitialPosition"]["InitialPosition"])[0]

def get_source_locations_from_config():
    env_parser = get_env_config()
    return parse_Vector3r_from_string(env_parser['SourceLocations']['SourceLocations'])

def get_max_simulation_steps_from_config():
    env_parser = get_env_config()
    return parse_Vector3r_from_string(env_parser['SimulationParams']['MaxTimeSteps'])

def get_sensor_model_params_from_config(agent_number):
    agent_parser = get_agent_config(agent_number)
    return BinarySensorParameters(float(agent_parser['SENSOR_MODEL_PARAMETERS']['FalsePositiveRate']), 
                                  float(agent_parser['SENSOR_MODEL_PARAMETERS']['FalseNegativeRate']))
    
def get_SPRT_params_from_config(agent_number):
    '''
    Returns the type1 and type2 probabilities of error
    '''
    agent_parser = get_agent_config(agent_number)
    return (float(agent_parser['SPRTParameters']['Type1ErrorProb']), float(agent_parser['SPRTParameters']['Type2ErrorProb']))

def get_ffmpeg_file_loc():
    env_parser = get_env_config()
    return env_parser["FileLocations"]["FFMPEGBinaryLocation"]


#%%
if __name__ == '__main__':
#%%    
    assert parse_Vector3r_from_string(" Vector3r(0,0) ") == [Vector3r(0,0)]
    assert parse_Vector3r_from_string(" Vector3r(0,0) Vector3r(5,6)") == [Vector3r(0,0), Vector3r(5,6)]
    
    get_simulated_sensor_from_config(1)
    get_grid_from_config()
    get_agent_start_pos_from_config(1)
    get_source_locations_from_config()
    get_max_simulation_steps_from_config()
    get_sensor_model_params_from_config(1)
    
    
    
    
    
    

