# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 11:03:14 2018

@author: 13383861
"""
import sys
import random
import itertools
#update path so other modules can be imported
sys.path.append('..')
#from AirSimInterface.types import Vector3r
from Utils.Vector3r import Vector3r

class UE4GridFactory:
    def __init__(self, lng_spacing, lat_spacing, origin, x_lim=None, y_lim=None, no_x=None, no_y=None):
        self.lat_spacing = lat_spacing
        self.lng_spacing = lng_spacing
        self.origin = origin
        if x_lim and y_lim:
            self.x_lim, self.y_lim = x_lim, y_lim
            self.create_grid_with_limits()
        if no_x and no_y: 
            self.no_x, self.no_y = no_x, no_y
            self.create_grid_with_no_points()
        if not all([x_lim, y_lim]) and not all([no_x, no_y]):
            raise Exception('Either give a limit to the grid or an x and y spacing')
            
        
    def create_grid_with_limits(self):
        #two cases where self.x_lim/self.lng_spacing can be integer or float both covered by this - if integer then add one to make sure
        #full range covered
        #otherwise division is rounded up (int cast rounds down and adding one rounds up)
        self.no_x = int((self.x_lim - self.origin.x_val)/self.lng_spacing) + 1
        self.no_y = int((self.y_lim - self.origin.y_val)/self.lat_spacing) + 1
        self.create_grid_with_no_points()
        
    def create_grid_with_no_points(self):
        self.grid = []
        backtrack = False
        for x_counter in range(self.no_x):
            if backtrack:
                for y_counter in range(self.no_y):
                    self.grid.append(self.origin + Vector3r(x_counter * self.lng_spacing, y_counter * self.lat_spacing))
                backtrack = not backtrack
            else:
                for y_counter in range(self.no_y-1, -1, -1):
                    self.grid.append(self.origin + Vector3r(x_counter * self.lng_spacing, y_counter * self.lat_spacing))
                backtrack = not backtrack
            
    def get_grid_points(self):
        return self.grid
    
class UE4Grid:
    def __init__(self, lng_spacing, lat_spacing, origin, x_lim=None, y_lim=None, no_x=None, no_y=None):
        if not all([x_lim, y_lim]) and not all([no_x, no_y]):
            raise Exception('Either give a limit to the grid or an x and y spacing')
            
        self.origin = origin
        self.grid_factory = UE4GridFactory(lng_spacing, lat_spacing, origin, x_lim, y_lim, no_x, no_y)
        self.grid_points = self.grid_factory.get_grid_points()
        
        self.lat_spacing = self.grid_factory.lat_spacing
        self.lng_spacing = self.grid_factory.lng_spacing
        
        self.no_x = self.grid_factory.no_x
        self.no_y = self.grid_factory.no_y
    
    def get_grid_points(self):
        return self.grid_points
    
    def get_lat_spacing(self):
        return self.lat_spacing
    
    def get_lng_spacing(self):
        return self.lng_spacing
    
    def get_no_points_x(self):
        return self.no_x
    
    def get_no_points_y(self):
        return self.no_y
    
    def get_neighbors(self, grid_loc, radius):
        '''Gets neighbors of grid_loc within radius.'''
        return list(filter(lambda alt_grid_loc: alt_grid_loc.distance_to(grid_loc) <= radius and alt_grid_loc != grid_loc, self.get_grid_points()))
    
    def get_diameter(self):
        '''Returns the maximum distance between any two nodes in the grid'''
        #finds max value of a list of distances from every point to every other point
        return max([loc1.distance_to(loc2) for loc1, loc2 in itertools.permutations(self.grid_points, 2)])
    
#%%
class UE4GridWithSources(UE4Grid):
    '''For testing purposes, can query grid to see if source is there or not'''
    def __init__(self, lng_spacing, lat_spacing, origin, sources_locations, x_lim=None, y_lim=None, no_x=None, no_y=None):
        super().__init__(lng_spacing, lat_spacing, origin, x_lim, y_lim, no_x, no_y)
        self.sources_locations = sources_locations
        
    def rad_model(self, location, radiation_locs: 'list of simpleCoord'):
        '''Given a reading, returns probability of radiation at current location. Might be best to use a mixture model or similar
        Current approach assumes that the count data is additive for multiple sources.
        Then the probability of a detection is simply the sum of the counts divided by a normalizing factor which converts count to probability.
        '''
        #there will be some count data which is a function of distance. If this count data is beyond a certain threshold, then 
        #radiation has been detected.
        
        #ionizing radition strength in micro sieverts per hous is sigma/d^2, where sigma is strength at a distance 1m form source
        sigma = 200
        strength = lambda d: sigma/d**2 if d != 0 else 1000
        #this assumes that the count at a point in between multiple radiation sources equals the sum of the counts of the 
        #individual sources at that point.
        strengths = [strength(location.distance_to(radiation_loc)) for radiation_loc in radiation_locs]
        #returns true if within 0.5 m
        return 1 if sum(strengths)/800 > 0.95 else ((sum(strengths)/800 + 0.1)*0.9) + random.gauss(0,0.005)

    def get_sensor_reading(self):
        return self.rad_model(self.current_pos_intended, self.sources_locations)

#%%
    
if __name__ == "__main__":
    
    test_grid = UE4Grid(2, 1, Vector3r(0,0), 10, 6)
    
    assert test_grid.get_no_points_x() == 6
    assert test_grid.get_no_points_y() == 7
    
    assert test_grid.get_lat_spacing() == 1
    assert test_grid.get_lng_spacing() == 2
    
    assert len(test_grid.get_grid_points()) == test_grid.get_no_points_x() * test_grid.get_no_points_y()
    
    test_grid1 = UE4Grid(1, 1, Vector3r(0,0), 9, 13)
    assert len(test_grid1.get_grid_points()) == test_grid1.get_no_points_x() * test_grid1.get_no_points_y()    
    
    assert set(test_grid1.get_neighbors(Vector3r(2,2), 1.9)) == set([Vector3r(1,2), Vector3r(2,1), Vector3r(2,3), Vector3r(3,2), Vector3r(3,3), Vector3r(1,3), Vector3r(1,1), Vector3r(3,1)])
    assert set(test_grid.get_neighbors(Vector3r(2,2), 2)) ==  set([Vector3r(0,2), Vector3r(2,3), Vector3r(2,4),Vector3r(4,2),Vector3r(2,0),Vector3r(2,1)])
    
    assert test_grid.get_diameter() == Vector3r(0,0).distance_to(Vector3r(10,6))
    
    test_grid2 = UE4Grid(1, 1, Vector3r(6,3), 9, 13)
    test_grid2.get_grid_points()
    assert test_grid2.get_diameter() == Vector3r(6,3).distance_to(Vector3r(9,13))
    
    
    
    
    
    
    
    