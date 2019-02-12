# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 19:30:42 2019

@author: 13383861
"""


#prior should be a map from typing.Dict[Vector3r, float]
import typing

from scipy.stats import multivariate_normal
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from Utils.UE4Grid import UE4Grid
from Utils.Vector3r import Vector3r
#for testing purposes


#%%
class Prior:
    def __init__(self, grid, distribution_generator: "A function that takes in a grid location and returns the corresponding prior pdf value"):
        self.prior_dict = {grid_loc: distribution_generator(grid_loc) for grid_loc in grid.get_grid_points()}
        
    def get_prior_dict(self):
        return self.prior_dict
        
    def show_prior(self):
        '''Shows prior on 3d graph'''
        pass
    
def _generate_gaussian_prior(grid: UE4Grid, means: "list of 2 means", covariance_matrix: "2x2 covariance matrix", initial_belief_sum = 0.5) -> "Tuple(np.array normed prior probs, prior_dict":
    '''A method that returns the normed z vector as well as the prior dict. Given a 2d vector of means, 2X2 covariance matrix, returns a 2D gaussian prior'''
    #maybe should check all dimensions of data here
    prior = {}
    x, y = np.mgrid[0:grid.get_no_points_x() * grid.get_lng_spacing():grid.get_lng_spacing(), 0:grid.get_no_points_y() * grid.get_lat_spacing():grid.get_lat_spacing()]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y
    rv = multivariate_normal(means, covariance_matrix)
    z = rv.pdf(pos)
    normed_z = z * (initial_belief_sum/z.sum())
    
    #normed_z = normed_z.astype(float)
    #return likelihoods in dict with grid points
    for x_value in [_[0] for _ in x]:
        for y_value in y[0]:
            prior[Vector3r(x_value, y_value)] = float(normed_z[x_value][y_value])
    return normed_z, prior
    
def generate_gaussian_prior(grid: UE4Grid, means: "list of 2 means", covariance_matrix: "2x2 covariance matrix", initial_belief_sum = 0.5) -> "Tuple(np.array normed prior probs, prior_dict":
    '''Given a 2d vector of means, 2X2 covariance matrix, returns a 2D gaussian prior'''
    return _generate_gaussian_prior(grid, means, covariance_matrix, initial_belief_sum)[1]


def generate_uniform_prior(grid: UE4Grid, initial_belief_sum = 0.5, fixed_value = None):
    '''Generates a uniform prior which equates to delta/|grid|. The sum of each grid location prior is equal to initial_belief_sum. If fixed_value provided, 
    each location in the prior will have fixed_value as prior'''
    if fixed_value and 0<=fixed_value<=1:
        prior_val = fixed_value
    else:
        prior_val = initial_belief_sum/len(grid.get_grid_points())
    return {grid_loc: prior_val for grid_loc in grid.get_grid_points()}
    
def plot_gaussian_prior(grid: UE4Grid, means: "list of 2 means", covariance_matrix: "2x2 covariance matrix", initial_belief_sum = 0.5):
    prior = _generate_gaussian_prior(grid, means, covariance_matrix, initial_belief_sum)[0]
    fig = plt.figure()
    x, y = np.mgrid[0:grid.get_no_points_x() * grid.get_lng_spacing():grid.get_lng_spacing(), 0:grid.get_no_points_y() * grid.get_lat_spacing():grid.get_lat_spacing()]
    ax = fig.gca(projection='3d')
    z_lim = prior.max() * 1.02
    ax.set_zlim3d(0, z_lim)
    ax.plot_wireframe(x, y, prior)
    
def save_gaussian_prior(grid: UE4Grid, means: "list of 2 means", covariance_matrix: "2x2 covariance matrix", file_path, initial_belief_sum = 0.5):
    prior = _generate_gaussian_prior(grid, means, covariance_matrix, initial_belief_sum)[0]
    fig = plt.figure()
    x, y = np.mgrid[0:grid.get_no_points_x() * grid.get_lng_spacing():grid.get_lng_spacing(), 0:grid.get_no_points_y() * grid.get_lat_spacing():grid.get_lat_spacing()]
    ax = fig.gca(projection='3d')
    z_lim = prior.max() * 1.02
    ax.set_zlim3d(0, z_lim)
    ax.plot_wireframe(x, y, prior)
    plt.savefig(file_path)
    
#%%
if __name__ == '__main__':

    from Utils.BeliefMap import BeliefMapComponent, create_single_source_belief_map, create_belief_map
    grid = UE4Grid(1, 1, Vector3r(0,0), 10, 10)
    grid.get_grid_points()
    means = [1,3]
    covariance_matrix = [[7.0, 0], [0, 15]]
    initial_belief_sum = 0.5
    prior = _generate_gaussian_prior(grid, means, covariance_matrix, initial_belief_sum)
    
    fig = plt.figure()
    x, y = np.mgrid[0:grid.get_no_points_x() * grid.get_lng_spacing():grid.get_lng_spacing(), 0:grid.get_no_points_y() * grid.get_lat_spacing():grid.get_lat_spacing()]
    ax = fig.gca(projection='3d')
    ax.set_zlim3d(0, 0.02)
    ax.plot_wireframe(x, y, prior[0])
    
    plot_gaussian_prior(grid, means, covariance_matrix)
    
    prior = generate_gaussian_prior(grid, means, covariance_matrix, initial_belief_sum = 0.5)
    assert 0.49 < sum(prior.values()) < 0.51
    
    vectors = list(prior.keys())
    assert type(vectors[0].x_val) == float
    
    
    uniform_prior = generate_uniform_prior(grid, initial_belief_sum = 0.5)
    assert 0.49 < sum(uniform_prior.values()) < 0.51
    assert all([prior_val == 0.5/121 for prior_val in list(uniform_prior.values())])

#    print(type(prior))
#    Vector3r(0, 10, 0.0) in prior
#    prior[Vector3r(0.0, 10.0, 0.0)]
#    grid_point = grid.get_grid_points()[0]
#    type(list(prior.keys())[0].x_val)
#    list(prior.keys())[0]._verify_floats()
#    
#    [grid_point for grid_point in grid.get_grid_points()]
#    BeliefMapComponent(grid_point, float(prior[grid_point])) 
#    
#    [BeliefMapComponent(grid_point, float(prior[grid_point])) for grid_point in grid.get_grid_points()]

    single_source_bel_map = create_single_source_belief_map(grid, "agent1", prior = prior, alpha = 0.2, beta = 0.1)
    multiple_source_bel_map = create_belief_map(grid, "agent1", prior = prior)


