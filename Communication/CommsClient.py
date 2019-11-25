# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 15:37:32 2019

@author: 13383861
"""

import requests
from Utils.Vector3r import Vector3r
from Utils.AgentObservation import AgentObservation, BinaryAgentObservation

class AgentCommunicatorClient:
    
    base_port = 8000
    request_string = "/get_observations/{other_agent_name}/{start_timestep}/{end_timestep}"
    
    def create_request_url(self, other_agent_name, start_timestep = '0', end_timestep = None):
        return "http://127.0.0.1:" + str(AgentCommunicatorClient.base_port + int(other_agent_name.split('agent')[1])) + AgentCommunicatorClient.request_string.format(other_agent_name =other_agent_name ,  start_timestep = start_timestep, end_timestep = end_timestep)
    
    def get_observations_from(self, other_agent_name, start_timestep = '0', end_timestep = None):
        '''
        Returns a list of agent observations from another agent given the other agents name.
        '''
        #print("making get request to: ", self.create_request_url(other_agent_name, timestamp))
        response_json = requests.get(self.create_request_url(other_agent_name, start_timestep, end_timestep)).json()
        return_observations = []
        for row in response_json[1:]:            
            row = row.split(',')
#            try:
            return_observations.append(BinaryAgentObservation(Vector3r(row[1], row[0], row[2]), *row[3:]))
#           except Exception as e:
#                print("found an exception: ", row)
        return return_observations
        #return response
        
    def shutdown_server(self, agent_name):
        requests.get("http://127.0.0.1:" + str(AgentCommunicatorClient.base_port + int(agent_name.split('agent')[1])) + "/shutdown")
        
    def check_server_running(self, agent_name):
        try:
            return requests.get("http://127.0.0.1:" + str(AgentCommunicatorClient.base_port + int(agent_name.split('agent')[1])) + '/ping').json() == 'Successfully pinged'
        except Exception:
            return False
        
        
        
        
        