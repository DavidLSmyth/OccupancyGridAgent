# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 11:08:36 2019

@author: 13383861
"""

# A class that sets up a simple grid for testing purposes.
import sys
import random
import math
import scipy.stats
#update path so other modules can be imported
sys.path.append('..')
from Utils.UE4Grid import UE4Grid


class SimpleCoord:
    def __init__(self, x, y):
        self.x_val = x
        self.y_val = y
    
    def manhattan_dist(self, other: 'SimpleCoord'):
        return abs((self.x_val- other.x_val) + (self.y_val- other.y_val))
    
    def pythagorean_dist(self, other: 'SimpleCoord'):
        return ((self.x_val- other.x_val)**2 + (self.y_val- other.y_val)**2)**0.5
    
    def __str__(self):
        return f"({self.x_val}, {self.y_val})"
    
    def __repr__(self):
        return f"({self.x_val}, {self.y_val})"
    
    
    def __eq__(self, other):
        return self.x_val== other.x_val and self.y_val== other.y_val
    
    def __hash__(self):
        return hash(str(self.x_val) + str(self.y_val))


##run some simple tests
#c1 = SimpleCoord(2,2)
#c2 = SimpleCoord(5,6)
#assert c1.manhattan_dist(c2) == 7
#assert c1.pythagorean_dist(c2) == 5
#print(c1)

def rad_model(location: SimpleCoord, radiation_locs: 'list of simpleCoord'):
    '''Given a reading, returns probability of radiation at current location'''
    #there will be some count data which is a function of distance. If this count data is beyond a certain threshold, then 
    #radiation has been detected.
    
    #ionizing radition strength in micro sieverts per hous is sigma/d^2, where sigma is strength at a distance 1m form source
    sigma = 200
    strength = lambda d: sigma/d**2 if d != 0 else 1000
    strengths = [strength(location.pythagorean_dist(radiation_loc)) for radiation_loc in radiation_locs]
    #returns true if within 0.5 m
    return 1 if sum(strengths)/800 > 0.95 else ((sum(strengths)/800 + 0.1)*0.9) + random.gauss(0,0.005)
    
#print(rad_model(SimpleCoord(0,0), [SimpleCoord(0.5,0)]))
#print(rad_model(SimpleCoord(2,2), [SimpleCoord(3,2), SimpleCoord(4.3,5.1)]))
#print(rad_model(SimpleCoord(2,2.5), [SimpleCoord(3,2), SimpleCoord(4.3,5.1)]))
#print(rad_model(SimpleCoord(1,1), [SimpleCoord(3,2), SimpleCoord(4.3,5.1)]))
#grid_ = [SimpleCoord(a/10,b/10) for a in range(0,50,1) for b in range(0,50,1)]
#rad_1 = SimpleCoord(1,1)
#rad_2 = SimpleCoord(4,4.7)
#radiation_locs = [rad_1, rad_2]
#readings = [rad_model(loc, radiation_locs) for loc in grid_]
#
#
#from mpl_toolkits.mplot3d import Axes3D # This import has side effects required for the kwarg projection='3d' in the call to fig.add_subplot
#
#import numpy as np
#import matplotlib
#matplotlib.use('TkAgg')
#import numpy as np
#import matplotlib.pyplot as plt
#
#from mpl_toolkits import mplot3d
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#X,Y = np.meshgrid(list(map(lambda coord: coord.x_val, grid_)), list(map(lambda coord: coord.y_val, grid_)))
#X = np.array(list(map(lambda coord: coord.x_val, grid_))).reshape([50,50])
#Y = np.array(list(map(lambda coord: coord.y_val, grid_))).reshape([50,50])
#Z = np.array(readings).reshape([50,50])
##ax.plot3D(, 'gray')
#ax.plot_surface(X,Y, Z)
#fig.show()
#plt.show()



class SimpleGrid:
    
    
    '''A simple grid to be used for testing purposes for a simple grid agent. There could potentially be multiple sources of radiation, so 
    The approach taken is that grid cells are independent of each other as in "Coordinated Search with a Swarm of UAVs". 
    In future joint distribution can be taken into account using Thruns "Learning Occupancy Grids with Forward Models.''' 
    
    
    def __init__(self, x_spacing, y_spacing, source_epicenters, effective_range: "effective range of the sensor, a distance", no_x=None, no_y=None):
        self.grid_points = [SimpleCoord(x * x_spacing , y * y_spacing) for x in range(no_x) for y in range(no_y)]
        self.x_spacing = x_spacing
        self.y_spacing = y_spacing
        self.no_x = no_x
        self.no_y = no_y
        self.source_epicenters = source_epicenters
        self.effective_range = effective_range
    
    def get_grid_points(self):
        return self.grid_points
    
    def get_x_spacing(self):
        return self.x_spacing
    
    def get_y_spacing(self):
        return self.y_spacing
    
    def get_no_points_x(self):
        return self.no_x
    
    def get_no_points_y(self):
        return self.no_y
    
    def get_neighbors(self, grid_loc, radius):
        '''Gets neighbors of grid_loc within radius.'''
        return list(filter(lambda alt_grid_loc: alt_grid_loc.pythagorean_dist(grid_loc) <= radius and alt_grid_loc != grid_loc, self.get_grid_points()))
    
    def get_reading(self, grid_loc):
        return rad_model(grid_loc, self.source_epicenters)
    
if __name__ == "__main__":
    test_grid = SimpleGrid(1, 1, [SimpleCoord(4,5), SimpleCoord(3,5)], 2, 5, 5)
    print(test_grid.get_reading(SimpleCoord(4,4.5)))
    
    assert test_grid.get_reading(SimpleCoord(4,5)) >= 0.4
    assert test_grid.get_reading(SimpleCoord(1,5)) >= 0.4
    assert len(test_grid.get_grid_points()) == 25
    for i in set(test_grid.get_neighbors(SimpleCoord(2,2), 1.5)):
        print(i)
    assert set(test_grid.get_neighbors(SimpleCoord(2,2), 1.5)).difference(set([SimpleCoord(1,3), SimpleCoord(2,3),SimpleCoord(3,3), SimpleCoord(1,2), SimpleCoord(3,2), SimpleCoord(1,1), SimpleCoord(2,1), SimpleCoord(3,1)])) == set()

    