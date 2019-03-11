# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 14:46:56 2019

@author: 13383861
"""

import matplotlib.pyplot as plt
import requests
import sys
import random
import os
sys.path.append('.')

#import AirSimInterface.client as airsim
#from AirSimInterface.types import ImageRequest, ImageType, Vector3r
from Utils.Vector3r import Vector3r

class RadModel:
    '''Only model count data - if above certain threshold, sensor should return 1. 
    Maybe could add a plume prediction model to this.'''    
    def __init__(self, radiation_locs, grid, sigma):
        self.radiation_locs = radiation_locs
        self.grid = grid
        self.sigma = sigma
        self.strength = lambda d: self.sigma/d**2 if d >= 0.1 else sigma/0.1
            
    def get_dose(self, location):
        '''Take dose and count to both follow inverse square law for now'''
        #add a random component proportional to sigma
        #the random component introduces noise to the data - change the variance to 
        strengths = [self.strength(location.distance_to(radiation_loc)) + self.sigma * random.gauss(0,0.0002) for radiation_loc in self.radiation_locs]
        strengths  = [0 if strength < 0 else strength for strength in strengths]
        return sum(strengths)        
#        if sum(strengths)/800 > 0.95:
#            return (random.random() * 0.05)+0.95
#        return 1 if sum(strengths)/800 > 0.95 else ((sum(strengths)/800 + 0.1)*0.9) + random.gauss(0,0.005)
    
    def plot_falloff(self, filepath):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        X = list(map(lambda coord: coord.x_val, self.grid.get_grid_points()))
        Y = list(map(lambda coord: coord.y_val, self.grid.get_grid_points()))
        print(X)
        print(Y)
        Z = [self.get_dose(grid_point) for grid_point in self.grid.get_grid_points()]
        ax.plot_trisurf(X, Y, Z)
        plt.savefig(filepath)
        
        
class RadSensor:
    
    def __init__(self,rad_model, sensitivity: 'sensitivity is the radius within which sensor would pick up "high" radiation'):
        self.rad_model = rad_model
        self.sensitivity = sensitivity
    
    def get_probability(self, location):
        '''Need to pass in location to the model, in real life, will just send a query to the rad sensor to return a value
        based on the reading at the current agent location.'''
        dose = self.rad_model.get_dose(location)
        #if 0.1 m away return 1
        if dose >= self.rad_model.strength(self.sensitivity):
            return 1
        else:
            return dose/self.rad_model.strength(self.sensitivity)
    
    def plot_falloff(self, filepath):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        X = list(map(lambda coord: coord.x_val, self.rad_model.grid.get_grid_points()))
        Y = list(map(lambda coord: coord.y_val, self.rad_model.grid.get_grid_points()))
        print(X)
        print(Y)
        Z = [self.get_probability(grid_point) for grid_point in self.rad_model.grid.get_grid_points()]
        ax.plot_trisurf(X, Y, Z)
        plt.savefig(filepath)
    
    
class AirsimImageSensor:
    
    def __init__(self, agent_name, image_dir, airsim_client):
        self.agent_name = agent_name
        self.image_dir = image_dir
        self.airsim_client = airsim_client
        
    def _get_image_response(self, image_loc: str):
        '''Queries microsoft NN for most likely class, reponse is returned as JSON'''
        headers = {'Prediction-Key': "fdc828690c3843fe8dc65e532d506d7e", "Content-type": "application/octet-stream", "Content-Length": "1000"}
        with open(image_loc,'rb') as f:
            response =requests.post('https://southcentralus.api.cognitive.microsoft.com/customvision/v2.0/Prediction/287a5a82-272d-45f3-be6a-98bdeba3454c/image?iterationId=3d1fd99c-0b93-432f-b275-77260edc46d3', data=f, headers=headers)
        return response.json()
    
    def _record_Airsim_image(self, airsim_client):
        '''Assumes that airsim_client has moved to correct position. Image is recorded in image_dir/photoxxyyy.png'''
        responses = self.airsim_client.simGetImages([ImageRequest("3", ImageType.Scene)], vehicle_name = self.agent_name)
        response = responses.pop()
        # get numpy array
        #filename = OccupancyGridAgent.ImageDir + "/photo_" + str(self.timestep)
        filename = self.image_dir + "/photo_" + str(self.timestep)
        airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8) 
        return os.path.normpath(filename + '.png')
    
    def _get_highest_pred(self, image_json):
        '''Given image json returned by microsoft vision ai, detemines the highest prediction'''
        max_pred = 0
        max_pred_details = ''
        for pred in image_json['predictions']:
            if pred['probability'] > max_pred:
                max_pred = pred['probability']
                max_pred_details = pred
        return max_pred, max_pred_details
        
        
    def get_probability(self, location):
        '''Takes an image at current location and saves it. location is not necessary as a parameter but '''
        #assumes that rav has moved to location
        recorded_image_loc = self._record_Airsim_image()
        max_pred, max_pred_details = self._get_highest_pred(self._get_image_response(recorded_image_loc))
        return max_pred


class ChungBurdickSingleSourceSensor:
    '''A sensor which returns the probability of detection at a location, given the knowledge that there is a single source present'''
    def __init__(self, false_positive_rate, false_negative_rate, source_location):
        self.false_positive_rate = false_positive_rate
        self.false_negative_rate = false_negative_rate
        self.source_location = source_location
        
    def get_detection_val(self,location):
        rand_no = random.random()
        if location == self.source_location:
            #0 reading (false negative) generated with probability beta at grid location where source actually lies
            if rand_no < self.false_negative_rate:
                return 0
            else:
                return 1
        else:
            #1 reading (false positive) generated with probability alpha at grid location where source is not actually present
            if rand_no < self.false_positive_rate:
                return 1
            else:
                return 0
        
    def get_probability(self, location):
#        '''Returns probability of detection'''
#        detection_val = self.get_detection_val(self.location)
#        if detection_val == 1:
#            numerator = (1-self.beta)
#            denominator = 
#        
        return self.get_detection_val(location)    
    
class MultipleSourceSensor:
    '''A sensor which returns the probability of detection at a location, given the knowledge that there is a single source present'''
    def __init__(self, false_positive_rate, false_negative_rate, source_locations: list):
        if not isinstance(source_locations, list):
            raise Exception("Please provide a list of source_locations")
        if not all([isinstance(source_location, Vector3r) for source_location in source_locations]):
            raise Exception("Please provide a list of source_locations instead of {}".format(source_locations[0]))
        self.false_positive_rate = false_positive_rate
        self.false_negative_rate = false_negative_rate
        self.source_locations = source_locations
        print(source_locations)
        
    def get_detection_val(self,location):
        rand_no = random.random()
        if location in self.source_locations:
            print("should be postive")
            #0 reading (false negative) generated with probability beta at grid location where source actually lies
            if rand_no < self.false_negative_rate:
                return 0
            else:
                return 1
        else:
            #1 reading (false positive) generated with probability alpha at grid location where source is not actually present
            if rand_no < self.false_positive_rate:
                return 1
            else:
                return 0
        
    def get_probability(self, location):
#        '''Returns probability of detection'''
#        detection_val = self.get_detection_val(self.location)
#        if detection_val == 1:
#            numerator = (1-self.beta)
#            denominator = 
#        
        return self.get_detection_val(location)  
    
#
#def get_image_response(image_loc: str):
#    headers = {'Prediction-Key': "fdc828690c3843fe8dc65e532d506d7e", "Content-type": "application/octet-stream", "Content-Length": "1000"}
#    with open(image_loc,'rb') as f:
#        response =requests.post('https://southcentralus.api.cognitive.microsoft.com/customvision/v2.0/Prediction/287a5a82-272d-45f3-be6a-98bdeba3454c/image?iterationId=3d1fd99c-0b93-432f-b275-77260edc46d3', data=f, headers=headers)
#    return response.json()
#
#
#def get_highest_pred(image_json):
#    max_pred = 0
#    max_pred_details = ''
#    for pred in image_json['predictions']:
#        if pred['probability'] > max_pred:
#            max_pred = pred['probability']
#            max_pred_details = pred
#    return max_pred, max_pred_details
#        
#sensor_reading = lambda image_loc: get_highest_pred(get_image_response(image_loc))
#
#%%
if __name__ == '__main__':
    
    #%%
    import math
    
    #"false alarm rate"
    fpr = 0.2
    #"missed detection rate"
    fnr = 0.25
    
    source_location = Vector3r(2,2)
    cb_sensor = ChungBurdickSingleSourceSensor(fpr, fnr, source_location)

    no_samples = 100000
    
    #check fpr
    assert math.isclose(sum([cb_sensor.get_probability(Vector3r(0,0)) for i in range(no_samples)]), no_samples*fpr, rel_tol = 0.01)
    #check fnr
    assert math.isclose(sum([1-cb_sensor.get_probability(source_location) for i in range(no_samples)]), no_samples*(fnr), rel_tol = 0.01)











