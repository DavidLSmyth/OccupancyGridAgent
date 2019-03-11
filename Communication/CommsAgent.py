# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 11:17:27 2019

@author: 13383861
"""
from flask import Flask
from flask_restful import Resource, Api
import sys
sys.path.append('.')
sys.path.append('..')
from Utils.ObservationSetManager import AgentObservationFileManager

'''A class that provides functionality for agents to communicate with each other. Approach is RESTful to fit in with ROCSAFE architecture.
In ROCSAFE, POST requests will be sent for the most recent sensor reading. Simulate this here.

'''



data = {'agent1', 'bye'}        

class AgentCommunicator():
    '''A simple class which acts as a server for other agents wanting to communicate with this agent'''
    no_communicators = 0
    base_port = 8000
    
    class _AgentObservationsRequest(Resource):
        '''If a request comes in for observations from a single agent, this returns it'''
        def __init__(self):
            #a bit of a hack in order to avoid having to hard code in agent name - initialise the the obs_file_manager as none
            #and then initialise properly when request comes in
            self.obs_file_manager = None
            
        def get(self, agent_name, timestep):
            if not self.obs_file_manager:
                self.obs_file_manager = AgentObservationFileManager(agent_name, 'Observations/'+agent_name + '.csv')
            return self.obs_file_manager.read_all_observations_from_file()
            #return "Will give you back observations from agent {} as far as timestep {}".format(agent_name, timestep)
        
        def get_agent_data(agent_id):
            '''Reads agent data from a file and returns the csv in string format'''
            return data[agent_id]
        
    def __init__(self, agent_name, data_file_path, ip = '127.0.0.1', port = None):
        AgentCommunicator.no_communicators += 1
        if not port:
            port = AgentCommunicator.base_port + AgentCommunicator.no_communicators
        self.app = Flask(__name__)
        self.api = Api(self.app)
        self.api.add_resource(AgentCommunicator._AgentObservationsRequest, '/get_observations/<agent_name>/<timestep>')
        self.port = port
    
    def run_app(self):
        '''sets up the rest service for a given agent'''
        self.app.run(debug = True, port = self.port)  
    
    
def main():
    comm1 = AgentCommunicator("agent1", "../Observations/agent1.csv").run_app()
    #comm2 = AgentCommunicator("agent2", "../Observations/agent2.csv")

        
if __name__ == '__main__':
    main()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    