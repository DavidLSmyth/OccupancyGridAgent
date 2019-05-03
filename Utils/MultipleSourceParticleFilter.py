# -*- coding: utf-8 -*-
"""
Created on Thu May  2 13:35:37 2019

@author: 13383861
"""

import filterpy

import math
import time
import sys

import numpy as np
import numba



#a particle filter is unsuitable to stationary problems. The reason for this is that if the state containing a source is not present
#in the prior distritbution, since the motion model is stationary, the same set of particles will be present in the next step of 
#the algorithm. 

#%%
class ParticleFilter:
    
    '''A concrete implementation of a particle filter. The purpose is to maintain a set of particles which
    represent the probability of evidence sources existing at various locations. This implementation makes the following assumptions:
    
    1). State x is represented by the joint distribution <s1, s2, ..., sk, pos>, where 1 <= k <= n
    (i.e. assuming there could be between 0 and n sources present, with any combination of positions for each source)
    
    where si is the random variable representing the location of the ith source of evidence and pos is positition.
    It is not possible for si = sj but all other states are possible. s1 has support {0, 1, ..., n} where n is the 
    number of discrete locations in the occupancy grid.
    
    2). Transition is deterministic: source are stationary and position is detemined by control action 
    
    3). Senor model is characterised by parameters (a1, b1), (a2, b2), ..., (ak, bk) which represent fpr fnr
    for each possible piece of evidence. The idea is that it could be known that there are a few different types of
    object that the sensor can detect, with a varying fpr, fnr. E.g. it could detect small bags with one fpr, fnr
    and big bags with a different one. If it is known that there could be a maximum of 2 small bags and 3 big bags, 
    this should be taken into account by the detection model.
    
    '''
    
    def __init__(self, no_particles):
        self.no_particles = no_particles
        #this holds the particles fr the current timestep
        self.particles = np.array([])
        
        
    def update(self, sensor_reading, loc):
        '''Updates the set of particles for next time step'''
        pass
        
    def resample(self, particles, weights):
        pass
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        