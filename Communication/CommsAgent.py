# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 11:17:27 2019

@author: 13383861
"""
from flask import Flask
from flask_restful import Resource, Api

import socket
import requests
import argparse

'''A class that provides functionality for agents to communicate with each other. Approach is RESTful to fit in with ROCSAFE architecture.
In ROCSAFE, POST requests will be sent for the most recent sensor reading. Simulate this here.

'''


data = {'agent1', 'bye'}



class SimpleAgentRest():
    
    '''A bi-directional method of communication between agents'''
    
    class _SimpleAgentCommunicator(Resource):
        def get(self, data):
            return {"hello": "world"}
        
        def get_agent_data(agent_id):
            return data[agent_id]
        
    def __init__(self, ip = '127.0.0.1', port = 8000, data_file_path):
        self.app = Flask(__name__)
        self.api = Api(self.app)
        self.api.add_resource(SimpleAgentCommunicator._SimpleAgentCommunicator)
        self.port = port
        self.data_object = data_object
    
    def run_app(self):
        '''sets up the rest service for a given agent'''
        self.app.run(debug = True, port = self.port)  
    
    
def main():
    
    
        
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    