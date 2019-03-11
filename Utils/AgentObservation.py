# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 11:30:11 2018

@author: 13383861
"""

import typing
import csv
import sys
#update path so other modules can be imported
sys.path.append('..')
sys.path.append('.')

#from AirSimInterface.types import Vector3r
from Utils.Vector3r import Vector3r
from Utils.UE4Grid import UE4Grid

#an agent precept consists of a grid location, a detection probability, a timestep, a timestamp and the observer name
_AgentObservationBase = typing.NamedTuple('_AgentObservationBase', [('grid_loc', Vector3r),
                                                    ('probability', float),
                                                    ('timestep', int), 
                                                    ('timestamp', float), 
                                                    ('observer_name', str)])

#%%
class AgentObservation(_AgentObservationBase):
    '''A wrapper class of _AgentAnalysisState to enforce correct data types'''
    def __new__(cls, grid_loc,probability,timestep,timestamp,observer_name) -> '_AgentObservation':
        #does it make sense to just leave args in the constructor and let the return line handle an incorrect number of args
        args = (grid_loc,probability,timestep,timestamp,observer_name)
        #enforce correct data types
        return super(AgentObservation, cls).__new__(cls, *[d_type(value) if type(value) is not d_type else value for value, d_type in zip(args, _AgentObservationBase.__annotations__.values())])

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        else:
            return all([self.grid_loc == other.grid_loc, self.probability == other.probability, 
                        self.timestep == other.timestep, self.timestamp == other.timestamp, 
                        self.observer_name == other.observer_name])
            #return self.__dict__ == other.__dict__
        
    def __hash__(self):
        # this seems to be broken , might be easier to copy above
        #return hash((self.__dict__.values()))
        return hash((self.grid_loc, self.probability, 
                        self.timestep, self.timestamp, 
                        self.observer_name))
   
#%%    
class BinaryAgentObservation(_AgentObservationBase):
    '''A class that handles binary observations - as assumed in 
    A Decision-Making Framework for Control Strategies in Probabilistic Search'''
    def __new__(cls, grid_loc,binary_sensor_reading,timestep,timestamp,observer_name) -> '_AgentObservation':
        #does it make sense to just leave args in the constructor and let the return line handle an incorrect number of args
        if binary_sensor_reading not in [0,1]:
            raise Exception("Binary sensor reading not valid: {}".format(binary_sensor_reading))
        args = (grid_loc, binary_sensor_reading, timestep, timestamp, observer_name)
        return super().__new__(cls, *[d_type(value) if type(value) is not d_type else value for value, d_type in zip(args, _AgentObservationBase.__annotations__.values())])
    
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        else:
            return all([self.grid_loc == other.grid_loc, self.probability == other.probability, 
                        self.timestep == other.timestep, self.timestamp == other.timestamp, 
                        self.observer_name == other.observer_name])        
    def __hash__(self):
        return hash((self.__dict__.values()))
    
#%%
#Code and test for class which manages agent observations in a set grid
class AgentObservations():
    '''A class which records agent observations in a UE4Grid'''
    def __init__(self, grid: UE4Grid):
        self.grid = grid
        #observations consist of agent percepts
        self.observations = []
        
    def record_agent_observation(self, new_observation: AgentObservation):
        self.observations.append(new_observation)
        
    def get_most_recent_observation(self, observations = []):
        '''Returns the most recent observation in a list of observations'''
        if not observations:
            observations = self.observations
        return sorted(observations, key = lambda observation: observation.timestamp, reverse = True)[0]
    
    def get_most_recent_observation_at_position(self, grid_loc: Vector3r):
        return self.get_most_recent_observation(self.get_all_observations_at_position(grid_loc))
    
    def get_all_observations_at_position(self, grid_loc: Vector3r):
        return list(filter(lambda observation: observation.grid_loc == grid_loc, self.observations))
    
    
#%%
  
    
    
    
    
import functools
    
def _get_agent_observation_for_csv(grid_loc,probability,timestep,timestamp,observer_name):
    return functools.reduce(lambda x, y: str(x)+str(y), [grid_loc.y_val,grid_loc.x_val,grid_loc.z_val,probability,timestep,timestamp,observer_name])
     
def get_agent_observation_for_csv(agent_observation: AgentObservation):
    '''Returns elements of agent state that are important for analysis that can be written to csv. Position, battery cap., total_dist_travelled, battery_consumed, occ_grid'''
    #csv_headers = ['timestep', 'timestamp', 'rav_name', 'position_intended', 'position_measured', 'total_dist_travelled', 'remaining_batt_cap', 'prop_battery_cap_used', 'sensor_reading', 'occ_grid_locs', 'occ_grid_likelihoods', 'coordinated_with_other_bool', 'coordinated_with_other_names']
    #return str(agent_analysis_state._fields).replace(')','').replace('(','').replace("'", '')
    return _get_agent_observation_for_csv(**agent_observation._asdict())

def get_agent_observations_file_header():
    return "{grid_loc.y_val},{grid_loc.x_val},{grid_loc.z_val},{probability},{timestep},{timestamp},{observer_name}".replace('{','').replace('}','')

#AgentObservation = namedtuple('obs_location', ['grid_loc','probability','timestep', 'timestamp', 'observer_name'])
def _init_observations_file(file_path):
    with open(file_path, 'w+') as f:
        f.write(get_agent_observations_file_header())
        
#def _write_to_obserations_file(file_path, agent_observation):
#    with open(file_path, 'a') as f:
#        f.write('\n' + get_agent_observation_for_csv(agent_observation))
        
#writes a new observation to file_path(should be the file path of some agent's observation)
def _update_observations_file(file_path, agent_observation: AgentObservation):
    with open(file_path, 'a') as f:
        f.write('\n'+get_agent_observation_for_csv(agent_observation))
        
def get_agent_observations_from_file(file_path: str)->typing.List[AgentObservation]:
    '''Reads agent observations from a file and returns a list of agent observation objects'''
    #maybe have a type mapping for this
    try:
        with open(file_path) as f:
            reader = csv.reader(f)
            header = next(reader)
            return [AgentObservation(Vector3r(row[1], row[0], row[2]),*row[3:]) for row in reader]
    except Exception as e:
        return []
 
#%%
if __name__ == "__main__":
    
    #%%
    test_grid = UE4Grid(1, 1, Vector3r(0,0), 10, 6)
    test_agent_observations = AgentObservations(test_grid)
    
    obs1 = AgentObservation(Vector3r(0,0),0.5, 1, 1234, 'agent1')
    obs2 = AgentObservation(Vector3r(0,0),0.7, 2, 1235, 'agent1')
    obs3 = AgentObservation(Vector3r(0,1),0.9, 3, 1237, 'agent1')
    obs4 = AgentObservation(Vector3r(0,0),0.7, 2, 1235, 'agent1')
    
    obs5 = AgentObservation(Vector3r(1.0,2,0),0.4, 1, 1234, 'agent1')

#%%
    assert obs5.grid_loc == Vector3r(1,2,0)    
    
    #check eq method of agent observation
    assert not obs1.__eq__(obs2) 
    assert obs2.__eq__(obs4)
    assert not obs1.__eq__(obs4)
    
    #%%check that hash function correctly implemented
    assert obs1.__hash__() != obs2.__hash__()
    assert obs1.__hash__() != obs4.__hash__()
    assert obs2.__hash__() == obs4.__hash__()

#%%
    assert obs1.grid_loc == Vector3r(float(0), float(0))
    assert obs1.probability == 0.5
#%%    
    test_agent_observations.record_agent_observation(obs1)
    test_agent_observations.record_agent_observation(obs2)
    test_agent_observations.record_agent_observation(obs3)
    
    assert test_agent_observations.get_most_recent_observation() == obs3
    assert test_agent_observations.get_most_recent_observation_at_position(Vector3r(0,0)) == obs2
    assert test_agent_observations.get_all_observations_at_position(Vector3r(0,1)) == [obs3]
    assert get_agent_observations_file_header() == 'grid_loc.y_val,grid_loc.x_val,grid_loc.z_val,probability,timestep,timestamp,observer_name'
#%%    
    mock_data_fp = "D:/ReinforcementLearning/DetectSourceAgent/Data/MockData/testAgentObservation.csv"
    _init_observations_file(mock_data_fp)
    _update_observations_file(mock_data_fp, obs1)
    _update_observations_file(mock_data_fp, obs2)
    
    assert obs2 in read_agent_observations_for_analysis_file(mock_data_fp)
    assert obs1 in read_agent_observations_for_analysis_file(mock_data_fp)
    
    
    