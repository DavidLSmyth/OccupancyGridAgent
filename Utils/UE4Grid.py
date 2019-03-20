# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 11:03:14 2018

@author: 13383861
"""
import sys
import random
import itertools
import typing
from bisect import bisect_left
#update path so other modules can be imported
sys.path.append('..')
#from AirSimInterface.types import Vector3r
from Utils.Vector3r import Vector3r

import matplotlib.pyplot as plt
from matplotlib import colors

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
            
    def get_grid_points(self) -> typing.List[Vector3r]:
        '''Returns the list of grid points'''
        return self.grid
    
class UE4Grid:
    '''
    Constructor assumes that the grid is initialized with a regular spacing between points. A method is provided to remove points from
    the grid.
    '''
    def __init__(self, lng_spacing, lat_spacing, origin, x_lim=None, y_lim=None, no_x=None, no_y=None):
        if not all([x_lim, y_lim]) and not all([no_x, no_y]):
            raise Exception('Either give a limit to the grid or an x and y spacing')
            
        self.origin = origin
        self.grid_factory = UE4GridFactory(lng_spacing, lat_spacing, origin, x_lim, y_lim, no_x, no_y)
        
        #keep grid_points sorted so that updating is easy
        self.grid_points = sorted(self.grid_factory.get_grid_points(), key =  lambda grid_point: grid_point.x_val + grid_point.y_val * len(self.grid_factory.get_grid_points()))
        
        self.lat_spacing = self.grid_factory.lat_spacing
        self.lng_spacing = self.grid_factory.lng_spacing
        
        self.no_x = self.grid_factory.no_x
        self.no_y = self.grid_factory.no_y
    
    def get_grid_points(self):
        return self.grid_points
    
    
    #if the grid is not regular, these are not properly defined
    def get_lat_spacing(self):
        return self.lat_spacing
    
    def get_lng_spacing(self):
        return self.lng_spacing
    
    def get_no_points_x(self):
        return self.no_x
    
    def get_no_points_y(self):
        return self.no_y
    
    def get_neighbors(self, grid_loc, radius):
        '''
        Gets neighbors of grid_loc within radius. The distance metric is governed by the objects that make up the grid. If a distance_to
        method is not implmented, this method will fail.
        '''
        return list(filter(lambda alt_grid_loc: alt_grid_loc.distance_to(grid_loc) <= radius and alt_grid_loc != grid_loc, self.get_grid_points()))
    
    def get_diameter(self):
        '''Returns the maximum distance between any two nodes in the grid'''
        #finds max value of a list of distances from every point to every other point
        return max([loc1.distance_to(loc2) for loc1, loc2 in itertools.permutations(self.grid_points, 2)])
    
    def remove_grid_location(self, grid_point: Vector3r) -> bool:
        '''Removes a grid location from the grid. A grid location that is not already in the grid cannot be removed.
        Removal is done in-place, grid is updated if True returned, otherwise no modifications if false returned.'''
        if grid_point not in self.grid_points:
            return False
        else:
            self.grid_points.remove(grid_point)
            return True
        
    def show_geometric_grid(self):
        max_x, max_y = max(map(lambda grid_point: grid_point.x_val, self.grid_points)), max(map(lambda grid_point: grid_point.y_val, self.grid_points))
        min_x, min_y = min(map(lambda grid_point: grid_point.x_val, self.grid_points)), min(map(lambda grid_point: grid_point.y_val, self.grid_points))
        #data = [0 if ]
        x_vals = list(map(lambda grid_point: grid_point.x_val, self.grid_points))
        y_vals = list(map(lambda grid_point: grid_point.y_val, self.grid_points))
        plt.figure()
        plt.plot(x_vals, y_vals, 'ro')
        plt.xlim(0 if min_x > 0 else min_x, max_x)
        plt.ylim(0 if min_y> 0 else min_y, max_y)
        #fig.set_xticks(self.get_lng_spacing)
        #fig.set_yticks(self.get_lat_spacing)
        plt.grid(True)
        
        
    def add_grid_location(self, grid_point: Vector3r) -> bool:
        if not isinstance(grid_point, Vector3r):
            raise NotImplementedError("Can only add an object of type Vector3r to the grid")
        if not grid_point in self.grid_points:
            #binary search for where to add grid_point
            #indices recording where to insert
            insertion_indices= [g_point.x_val + (g_point.y_val * self.get_no_grid_points()) for g_point in self.grid_points]
            insertion_index = grid_point.x_val + (grid_point.y_val * self.get_no_grid_points())
            insertion_location = bisect_left(insertion_indices, insertion_index)
            self.grid_points.insert(insertion_location, grid_point)
            return True
        else:
            return False
        
    def get_no_grid_points(self):
        return len(self.grid_points)
        
    def remove_grid_locations(self, grid_locations: typing.List[Vector3r]):
        '''Removes grid locations if they are present in the grid. Returns a list of booleans
        representing if the ith grid location was removed'''
        return [self.remove_grid_location(grid_location) for grid_location in grid_locations]
            
    
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
    #%%
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
    
    
    test_grid2.show_geometric_grid()
    
    test_grid2.remove_grid_location(Vector3r(8,8))
    test_grid2.remove_grid_location(Vector3r(8,9))
    print(test_grid2.get_grid_points())
    test_grid2.show_geometric_grid()
    
    test_grid2.add_grid_location(Vector3r(11, 13))
    test_grid2.show_geometric_grid()

    