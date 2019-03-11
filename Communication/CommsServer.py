# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 11:17:27 2019

@author: 13383861
"""
from flask import Flask, request
from flask_restful import Resource, Api
#%%
import sys
sys.path.append('.')
sys.path.append('..')
from Utils.ObservationSetManager import AgentObservationFileReader

'''
A class that provides functionality for agents to communicate with each other. Approach is RESTful to fit in with ROCSAFE architecture.
In ROCSAFE, POST requests will be sent for the most recent sensor reading. This implementation uses a GET request for simplicity.
'''



#%%
class AgentCommunicatorServer:
    '''A simple class which acts as a server for other agents wanting to communicate with this agent'''
    
    no_communicators = 0
    base_port = 8000
    
    class _AgentObservationsRequest(Resource):
        '''If a request comes in for observations from a single agent, this returns it'''
        def __init__(self):
            #a bit of a hack in order to avoid having to hard code in agent name - initialise the the obs_file_manager as none
            #and then initialise properly when request comes in
            self.obs_file_reader = None
            
        def get(self, agent_name, timestep):
            if not self.obs_file_reader:
                self.obs_file_reader = AgentObservationFileReader(agent_name, 'Observations/'+agent_name + '.csv')
                observations = self.obs_file_reader.get_agent_observations_from_file_raw(timestep = int(timestep))
                print("read observations as {}".format(observations))
            return observations
            #return "Will give you back observations from agent {} as far as timestep {}".format(agent_name, timestep)
            
            
    class _ShutdownServer(Resource):
        '''Allows the user to request a shutdown of the flask server'''
        def shutdown(self):
            func = request.environ.get('werkzeug.server.shutdown')
            if func is None:
                raise RuntimeError('Not running with the Werkzeug Server')
            func()
            
        def get(self):
            self.shutdown()
            return 'Server shut down request submitted...'
        
        
    def __init__(self, agent_name, ip = '127.0.0.1'):
        AgentCommunicatorServer.no_communicators += 1
        port = AgentCommunicatorServer.base_port + int(agent_name.split('agent')[1])
        self.app = Flask(__name__)
        self.api = Api(self.app)
        self.api.add_resource(AgentCommunicatorServer._AgentObservationsRequest, '/get_observations/<agent_name>/<timestep>')
        self.api.add_resource(AgentCommunicatorServer._ShutdownServer, '/shutdown')
        self.port = port
    
    def run_app(self):
        '''sets up the rest service for a given agent'''
        self.app.run(debug = True, port = self.port)  
    
    
def main():
    agent_name = sys.argv[1]
    print("starting comms for {}".format(agent_name))
    AgentCommunicatorServer(agent_name).run_app()
            
if __name__ == '__main__':
    main()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    