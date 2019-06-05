# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 11:45:44 2018

@author: 13383861
"""

import sys
sys.path.append('..')
import typing
import functools
import csv
from Utils.AgentObservation import AgentObservation
#from AirSimInterface.types import Vector3r
from Utils.Vector3r import Vector3r
from Utils.UE4Grid import UE4Grid
from Utils.BeliefMap import (create_single_source_binary_belief_map,
                             create_multiple_source_binary_belief_map,
                             create_single_source_belief_map_from_observations,
                             create_multiple_source_belief_map_from_observations)

#%%
class AgentObservationFileReader:
    def __init__(self, agent_name, file_path = None):
        if not file_path:
            self.file_path = "Observations/" + agent_name.strip() + ".csv"
        else:
            self.file_path = file_path
        self._init_file_handle()
        
    def _init_file_handle(self):
        #open file for both reading and writing
        try:
            self.file_handle = open(self.file_path, 'r+')
        except FileNotFoundError as e:
            print("Consider using the convention to organise agent files")
            print(sys.path)
            raise e
        
    def get_str_all_observations_from_file(self):
        return str(self.read_all_observations_from_file())
        
    #this should probably be cached instead of being read often
    def read_all_observations_from_file(self):
        '''
        Returns all observations from file associated with agent as AgentObservations 
        '''
        try:
            self.file_handle.seek(0)
            reader = csv.reader(self.file_handle)
            #header = next(reader)
            #read the header and then throw it away
            next(reader)
            #reset the file handle to it's start position for reading again
            return [AgentObservation(Vector3r(row[1], row[0], row[2]),*row[3:]) for row in reader]
        #not sure how this could be handled for now
        except Exception as e:
            raise e
            
    def get_agent_observations_from_file_raw(self, timestep = None):
        '''
        Returns all observations from file associated with agent as a string. If timestep is provided then returns all observations before timestep
        '''
        self.file_handle.seek(0)
        if not timestep:
            return self.file_handle.readlines()
        else:
            all_lines = self.file_handle.readlines()
            #print([line.split(',')[4] for line in all_lines])
            return list(filter(lambda line: line.split(',')[4] != 'timestep' and int(line.split(',')[4]) <= timestep, all_lines))
        
#            reader = csv.reader(self.file_handle)
#            #header = next(reader)
#            #read the header and then throw it away
#            next(reader)
#            #reset the file handle to it's start position for reading again
#            return '\n'.join([','.join(row) for row in reader])

        #not sure how this could be handled for now

            
    def get_agent_observations_from_file(self) -> typing.List[AgentObservation]:
        '''
        Returns all observations recorded only by the agent that this manager corresponds to.
        '''
        return list(filter(lambda x: x.observer_name == self.agent_name, self.read_all_observations_from_file()))
    
    def get_agent_observation_file_path(self):
        return self.file_path
    
    
class AgentObservationFileWriter:
    
    '''
    A class that can read agent observations from a csv formatted file and write agent observations to a file. Each agent has its own observation file manager
    which records observations made by it (and other agent observations that have been communicated to it) up to time t. 
    '''    
   #maybe this should go in a config file
    agent_observation_file_header = "{grid_loc.y_val},{grid_loc.x_val},{grid_loc.z_val},{probability},{timestep},{timestamp},{observer_name}".replace('{','').replace('}','')
    def __init__(self, agent_name, file_path = None):
        if not file_path:
            self.file_path = "./Observations/" + agent_name.strip() + ".csv"
        else:
            self.file_path = file_path
        self._init_file_handle()
        self.init_file_header()
    
    def _init_file_handle(self):
        #open file for both reading and writing
        try:
            self.file_handle = open(self.file_path, 'r+')
        except FileNotFoundError as e:
            print("Consider using the convention to organise agent files")
            print(sys.path)
            raise e
            
    def init_file_header(self):
        self.file_handle.write(AgentObservationFileWriter.agent_observation_file_header)
    
    def update_file_with_observations(self, agent_observations):
        if not isinstance(agent_observations, list):
            raise Exception("Can only process a list of agent observations")
        else:
            for agent_observation in agent_observations:
                self.update_file_with_observation(agent_observation)
        
    def update_file_with_observation(self, agent_observation):
        if not isinstance(agent_observation, AgentObservation):
            raise Exception("Can only write object of type AgentObservation to file")
        else:
            self.file_handle.write(self._get_formatted_agent_observation_for_csv(agent_observation))
    
    def _get_formatted_agent_observation_for_csv(self, agent_observation):
        return "{grid_loc.y_val},{grid_loc.x_val},{grid_loc.z_val},{probability},{timestep},{timestamp},{observer_name}".format(**agent_observation._asdict())
    
    def get_agent_observation_file_path(self):
        return self.file_path        
    
    
#%%
class ObservationSetManager:
    
    '''
    Manages and stores the recieved of sensor measurements of other agents. Observations don't have to be taken at disrete locations - 
    the continuous position can be recorded and the grid location inferred from this.
    Agent belief at time t can be calculated from this set of observations.
    Observations are recorded in a csv file. 
    '''
    
    def __init__(self, agent_name: 'name of the agent that owns this observation list manager', observation_file_path: "The path at which agent observations will be stored" = None):

        #key value pairs of form agent_name: set of AgentObservations
        self.observation_sets = dict()
        self.agent_name = agent_name
        #agent should initialize its own observations
        self.init_rav_observation_set(self.agent_name)
        self.agent_observation_file_manager = AgentObservationFileWriter(agent_name, observation_file_path)
    
    #really strange behaviour: using this initialises the class with observations that don't exist... self.observation_sets[rav_name] = set()
    def init_rav_observation_set(self, rav_name, observations = None):
        '''initialise a new list of observations for an agent'''
        #if there are no observations, just initialize the agent_observations cprresponding to the agent name as an emtpy set
        if not observations:
            self.observation_sets[rav_name] = set()
        else:
            self.observation_sets[rav_name] = observations

    def get_all_observations(self):
        '''
        Reutrns all recorded observations from all agents
        '''
        return functools.reduce(lambda x,y: x.union(y) if x else y, self.observation_sets.values(), set())
    
    def get_all_observations_at_grid_location(self, grid_loc: Vector3r):
        '''
        Returns all agent observations made at a specified grid location
        '''
        return list(filter(lambda observation: observation.grid_loc == grid_loc, self.observations))
    
        
    def get_observation_set(self, rav_name) -> typing.Set[AgentObservation]:
        '''Get list of observations from a specific agent'''
        return self.observation_sets[rav_name]
    
    def update_with_observation(self, observation: AgentObservation):
        '''Update the observation set with an observation made from an agent'''
        if observation.observer_name not in self.observation_sets:
            self.init_rav_observation_set(observation.observer_name)
        #if observation not already included
        if observation not in self.observation_sets[observation.observer_name]:
            self.observation_sets[observation.observer_name].update(set([observation]))
            self.agent_observation_file_manager.update_file_with_observation(observation)
            return observation
        else:
            return None
    
    def update_agent_observation_set(self, observations: typing.Set[AgentObservation]):
        '''
        Updates the set of observations associated with an agent with potentially new observations
        '''
        for observation in observations:
            self.update_with_observation(observation)
        #check if rav is present before updating
        #if rav_name not in self.observation_sets:
        #    self.init_rav_observation_set(rav_name)
        #this avoids recording duplicate observations
        #self.observation_sets[rav_name].update(observations)
        
#    def get_belief_map_from_t1_to_t2(self, grid, initial_belief, t1, t2, agent_name, use_other_agent_observations = False):
#        '''
#        Returns the agent belief starting at a specified timestep and finishing at a second specified timestep. 
#        Uses all available agent observations between the two timesteps to update. This is useful in the case that
#        a source of evidence has been located and a second source of evidence is now being subsequently located.
#        '''
#        if use_other_agent_observations:
#            agents_for_belief_map = set(all(self.observation_sets.keys()))
#        else:
#            agents_for_belief_map = set([agent_name])
#            
#        observations_for_belief_map  = self._get_belief_map_using_specific_agent_observations_at_timestep(grid, prior, agents_for_belief_map, timestep, single_source) 
#            
        
#    def get_belief_map_at_timestep(self, grid, initial_belief, timestep, agent_name, single_source = False, use_other_agent_observations = False):
#        '''
#        Returns the agent belief/occupancy map at specified timestep. Can specify whether to use other agent observations or not.
#        Only really makes sense for a single source, since 
#        '''
#        if use_other_agent_observations:
#            agents_for_belief_map = set(all(self.observation_sets.keys()))
#        else:
#            agents_for_belief_map = set([agent_name])
#            
#        observations_for_belief_map  = self._get_belief_map_using_specific_agent_observations_at_timestep(grid, initial_belief, agents_for_belief_map, timestep, single_source) 
#        if single_source:
#            return create_single_source_belief_map_from_observations(grid, observations_for_belief_map, initial_belief)
#        else:
#            return create_multiple_source_belief_map_from_observations(grid, observations_for_belief_map, initial_belief)
        
    def _get_agent_observations_at_timestep(self, agent_names: "names of the agents whose observations will be used to create the belief map", timestep):
        '''
        Given a list of agent names, a timestep, returns the observations corresponding to those agents before the timestep.
        '''
        return list(filter(lambda observation: observation.timestep < timestep and observation.observer_name in agent_names, self.get_all_observations()))
      

    def update_from_other_obs_set_man(self, other):
        '''Might need to check that the timestamps must be different...'''
        for rav_name, observation_set in other.observation_sets.items():
            self.update_agent_obs_set(rav_name, observation_set)
            
    def get_discrete_belief_map_from_observations(self, grid):
        '''Given a descrete grid, returns a belief map containing the likelihood of the source
        being contained in each grid segment'''
        #ToDo:
        #Currently observations must be made at grid locations - instead compute which observations are made 
        #in each grid location and then compute the belief map
        return_belief_map = create_belief_map(grid, self.agent_name)
        return_belief_map.update_from_observations(self.get_all_observations())
        return return_belief_map
    
    def get_upper_confidence_interval_map_from_observations(self, grid, agent_name):
        return_CI_map = create_confidence_interval_map_from_observations(grid)
        return_CI_map.update_from_observations(self.get_observation_set(agent_name))
    
    def get_continuous_belief_map_from_observations(self, grid_bounds):
        '''Given grid bounds, returns a function which returns the likelihood given the 
        continuous position of the RAV. I.E. transform the discrete PDF as above to a 
        continuous one.'''
        pass

#%%


#%%
if __name__ == "__main__":
    
    test_grid = UE4Grid(1, 1, Vector3r(0,0), 6, 5)
    
    test_ObservationSetManager = ObservationSetManager('agent1')
    test_ObservationSetManager.observation_sets
    
    obs1 = AgentObservation(Vector3r(0,0),0.5, 1, 1234, 'agent2')
    obs2 = AgentObservation(Vector3r(0,0),0.7, 2, 1235, 'agent2')
    obs3 = AgentObservation(Vector3r(0,1),0.95, 3, 1237, 'agent2')
    obs4 = AgentObservation(Vector3r(0,1),0.9, 3, 1238, 'agent1')
    
    test_ObservationSetManager.init_rav_observation_set('agent2', set([obs1, obs2]))
    assert test_ObservationSetManager.get_observation_set('agent2') == set([obs1, obs2])
    
    test_ObservationSetManager.observation_sets
    test_ObservationSetManager.update_rav_obs_set('agent2', set([obs3]))
    
    test_ObservationSetManager.get_all_observations()
    
    assert test_ObservationSetManager.get_observation_set('agent2').intersection(set([obs1, obs2, obs3])) == set([]), "agent2 observations should have been added to set"
    assert test_ObservationSetManager.get_observation_set('agent1') == set([]), "agent1 observations should be empty"
       
    test_ObservationSetManager.update_with_obseravation(obs4)
    
    all_observations = test_ObservationSetManager.get_all_observations()
    assert not all_observations.difference(set([obs1, obs2, obs3, obs4])), "obs1, ..., obs4 should all be present in observation manager"

    
    #%%
    ###################################################
    # Check that duplicate observations aren't added
    test_grid = UE4Grid(1, 1, Vector3r(0,0), 6, 5)
    test1_ObservationSetManager = ObservationSetManager('agent1')
    
    obs1 = AgentObservation(Vector3r(0,0),0.5, 1, 1234, 'agent2')
    obs2 = AgentObservation(Vector3r(0,0),0.7, 2, 1235, 'agent2')
    obs3 = AgentObservation(Vector3r(0,1),0.95, 3, 1237, 'agent2')
    
    test1_ObservationSetManager.update_rav_obs_set('agent2',[obs1, obs2, obs3])
    test1_ObservationSetManager.observation_sets
    #test that duplicate measurements won't occur
    obs4 = AgentObservation(Vector3r(0,1),0.95, 3, 1237, 'agent2')
    test1_ObservationSetManager.update_rav_obs_set('agent2', set([obs4]))
    
    assert obs4 == obs3
    assert obs4.__hash__() == obs3.__hash__()
    
    assert obs4 != obs2
    assert obs4.__hash__() != obs2.__hash__()
    
    
    
    assert test1_ObservationSetManager.get_observation_set('agent2') == set([obs1, obs2, obs3]), "Duplicated observations should be ignored"
    
    assert abs(test1_ObservationSetManager.get_discrete_belief_map_from_observations(test_grid).get_belief_map_component(Vector3r(0,0)).likelihood - 0.074468) < 0.0001
    assert abs(test1_ObservationSetManager.get_discrete_belief_map_from_observations(test_grid).get_belief_map_component(Vector3r(0,1)).likelihood - 0.395833) < 0.0001
    
    
    
    